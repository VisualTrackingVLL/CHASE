import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from ltr import MultiGPU

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])

def _loss(data, actor):  # ,temp):
  loss,stats = actor(data)  # ,temp=temp) #forward into netwroks!!!!
  return loss

class Architect(object):

  def __init__(self, model,momentum_args, weight_decay_args,arch_learning_rate_args,arch_weight_decay_args):
    self.network_momentum = momentum_args
    self.network_weight_decay = weight_decay_args
    self.model = model.classifier.feature_extractor[0] #_loss,...->Just NAS
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

  def step(self,data, eta, network_optimizer, actor, unrolled): #network_optimizer: main optimizer containing all the parameters!! ->Double check -> May make ssome problems!
    data_t,data_v= data
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled([data_t,data_v], eta, network_optimizer,actor) #DARTS used train and validation for second-order approximation
    else:
        self._backward_step(data_v,actor) # DARTS used valid
    self.optimizer.step()

  def _backward_step(self, data_v,actor):
    loss = _loss(data_v,actor)#,self.temp)
    loss.backward()

  def _backward_step_unrolled(self, data,eta, network_optimizer,actor):
    data_t,data_v= data
    unrolled_model = self._compute_unrolled_model(data_t, eta, network_optimizer,actor)
    unrolled_loss = _loss(data_v,actor)#,self.temp) #Change

    unrolled_loss.backward()
    dalpha = [v.grad for v in unrolled_model._arch_parameters]
    vector = [v.grad.data for v in unrolled_model.parameters()]
    implicit_grads = self._hessian_vector_product(vector, data_t, actor)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

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
    loss = _loss(data_t,actor)#self.temp)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss = _loss(data_t,actor)#,self.temp)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]


