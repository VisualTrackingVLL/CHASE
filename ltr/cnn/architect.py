import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from ltr import MultiGPU

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])

def _loss(data, actor):
  loss,stats = actor(data)  #forward into netwroks!!!!
  return loss

class Architect(object):

  def __init__(self, model,momentum_args, weight_decay_args,arch_learning_rate_args,arch_weight_decay_args):
    self.network_momentum = momentum_args
    self.network_weight_decay = weight_decay_args
    self.model = model.neck #_loss,...->Just NAS
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(), #MixtureModel:self.model.classifier.feature_extractor[0]
        lr=arch_learning_rate_args, betas=(0.5, 0.999), weight_decay=arch_weight_decay_args)

  def _compute_unrolled_model(self,data_t, eta, network_optimizer,actor):
    loss = _loss(data_t,actor)#,self.temp)
    theta = _concat(self.model.parameters()).data
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    return unrolled_model

  def step(self,data, eta, network_optimizer, actor, unrolled): #network_optimizer: main optimizer containing all the parameters!!
    data_t,data_v= data
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled([data_t,data_v], eta, network_optimizer,actor)
    else:
        self._backward_step(data_v,actor)
    self.optimizer.step()

  def _backward_step(self, data_v,actor):
    loss = _loss(data_v,actor)
    loss.backward()

  def _backward_step_unrolled(self, data,eta, network_optimizer,actor):
    data_t,data_v= data
    unrolled_model = self._compute_unrolled_model(data_t, eta, network_optimizer,actor)
    train_feat = actor.net.extract_backbone_features(data_v['train_images'].reshape(-1, *data_v['train_images'].shape[-3:]))
    test_feat = actor.net.extract_backbone_features(data_v['test_images'].reshape(-1, *data_v['test_images'].shape[-3:]))
    train_temp=unrolled_model(train_feat)
    test_temp=unrolled_model(test_feat)
    target_scores = actor.net.classifier(train_temp, test_temp, data_v['train_anno'])
    train_feat_iou = actor.net.get_backbone_bbreg_feat(train_feat)
    test_feat_iou = actor.net.get_backbone_bbreg_feat(test_feat)
    iou_pred = actor.net.bb_regressor(train_feat_iou, test_feat_iou, data_v['train_anno'], data_v['test_proposals'])

    loss=self.compute_loss(target_scores,iou_pred,actor,data_v)
    loss.backward()

    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
    vector = [v.grad.data for v in unrolled_model.parameters()]
    implicit_grads = self._hessian_vector_product(vector, data_t, actor)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)


  def compute_loss(self,target_scores, iou_pred,actor,data):

    is_valid = data['test_anno'][:, :, 0] < 99999.0
    bb_scores = iou_pred[is_valid, :]
    proposal_density = data['proposal_density'][is_valid, :]
    gt_density = data['gt_density'][is_valid, :]

    # Compute loss
    bb_ce = actor.objective['bb_ce'](bb_scores, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)
    loss_bb_ce = actor.loss_weight['bb_ce'] * bb_ce

    # If standard DiMP classifier is used
    loss_target_classifier = 0
    loss_test_init_clf = 0
    loss_test_iter_clf = 0
    if 'test_clf' in actor.loss_weight.keys():
      # Classification losses for the different optimization iterations
      clf_losses_test = [actor.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in target_scores]

      # Loss of the final filter
      clf_loss_test = clf_losses_test[-1]
      loss_target_classifier = actor.loss_weight['test_clf'] * clf_loss_test

      # Loss for the initial filter iteration
      if 'test_init_clf' in actor.loss_weight.keys():
        loss_test_init_clf = actor.loss_weight['test_init_clf'] * clf_losses_test[0]

      # Loss for the intermediate filter iterations
      if 'test_iter_clf' in actor.loss_weight.keys():
        test_iter_weights = actor.loss_weight['test_iter_clf']
        if isinstance(test_iter_weights, list):
          loss_test_iter_clf = sum([a * b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
        else:
          loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

    # If PrDiMP classifier is used
    loss_clf_ce = 0
    loss_clf_ce_init = 0
    loss_clf_ce_iter = 0
    if 'clf_ce' in actor.loss_weight.keys():
      # Classification losses for the different optimization iterations
      clf_ce_losses = [actor.objective['clf_ce'](s, data['test_label_density'], grid_dim=(-2, -1)) for s in
                       target_scores]

      # Loss of the final filter
      clf_ce = clf_ce_losses[-1]
      loss_clf_ce = actor.loss_weight['clf_ce'] * clf_ce

      # Loss for the initial filter iteration
      if 'clf_ce_init' in actor.loss_weight.keys():
        loss_clf_ce_init = actor.loss_weight['clf_ce_init'] * clf_ce_losses[0]

      # Loss for the intermediate filter iterations
      if 'clf_ce_iter' in actor.loss_weight.keys() and len(clf_ce_losses) > 2:
        test_iter_weights = actor.loss_weight['clf_ce_iter']
        if isinstance(test_iter_weights, list):
          loss_clf_ce_iter = sum([a * b for a, b in zip(test_iter_weights, clf_ce_losses[1:-1])])
        else:
          loss_clf_ce_iter = (test_iter_weights / (len(clf_ce_losses) - 2)) * sum(clf_ce_losses[1:-1])

    # Total loss
    loss = loss_bb_ce + loss_clf_ce + loss_clf_ce_init + loss_clf_ce_iter + \
           loss_target_classifier + loss_test_init_clf + loss_test_iter_clf

    if torch.isinf(loss) or torch.isnan(loss):
      raise Exception('ERROR: Loss was nan or inf!!!')



    return loss

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, data_t, actor, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)
    loss = _loss(data_t,actor)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss = _loss(data_t,actor)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]


