import torch.optim as optim
from ltr.dataset import Lasot, Got10k, TrackingNet, MSCOCOSeq
from ltr.data import processing, sampler, LTRLoader
from ltr.models.tracking import dimpnet
import ltr.models.loss as ltr_losses
import ltr.models.loss.kl_regression as klreg_losses
import ltr.actors.tracking as tracking_actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU
import torch
import ltr.admin.settings as ws_settings

def run(settings):
    epochs=70
    start_epoch=0
    learning_rate_min=1e-4
    is_search= False
    settings.description = 'Default train settings for PrDiMP with ResNet50 as backbone.'
    settings.batch_size = 10
    settings.num_workers = 8
    settings.print_interval = 1
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 5.0
    settings.output_sigma_factor = 1/4
    settings.target_filter_sz = 4
    settings.feature_sz = 18
    settings.output_sz = settings.feature_sz * 16
    settings.center_jitter_factor = {'train': 3, 'test': 4.5}
    settings.scale_jitter_factor = {'train': 0.25, 'test': 0.5}
    settings.hinge_threshold = 0.05
    settings.print_stats = ['Loss/total', 'Loss/bb_ce', 'ClfTrain/clf_ce']

    # Train datasets
    if is_search:
        trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(4)))
        trackingnet_val = TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(5, 8)))
        got10k_test= Got10k(settings.env.got10k_dir, split='vottrain')
        lasot_test = Lasot(settings.env.lasot_dir, split='train')
        #coco_test = MSCOCOSeq(settings.env.coco_dir)
        got10k_val = Got10k(settings.env.got10k_dir, split='votval')

    else:
        lasot_train = Lasot(settings.env.lasot_dir, split='train')
        got10k_train = Got10k(settings.env.got10k_dir, split='vottrain')
        trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(8)))
        coco_train = MSCOCOSeq(settings.env.coco_dir)
        # Validation datasets
        got10k_val = Got10k(settings.env.got10k_dir, split='votval')
        trackingnet_val = TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(8,12)))

    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))
    if is_search:
        transform_val = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                  tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))
        transform_test = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))
    else:
        transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    # The tracking pairs processing module
    output_sigma = settings.output_sigma_factor / settings.search_area_factor
    proposal_params = {'boxes_per_frame': 128, 'gt_sigma': (0.05, 0.05), 'proposal_sigma': [(0.05, 0.05), (0.5, 0.5)]}
    label_params = {'feature_sz': settings.feature_sz, 'sigma_factor': output_sigma, 'kernel_sz': settings.target_filter_sz}
    label_density_params = {'feature_sz': settings.feature_sz, 'sigma_factor': output_sigma, 'kernel_sz': settings.target_filter_sz, 'normalize': True}

    data_processing_train = processing.KLDiMPProcessing(search_area_factor=settings.search_area_factor,
                                                        output_sz=settings.output_sz,
                                                        center_jitter_factor=settings.center_jitter_factor,
                                                        scale_jitter_factor=settings.scale_jitter_factor,
                                                        mode='sequence',
                                                        proposal_params=proposal_params,
                                                        label_function_params=label_params,
                                                        label_density_params=label_density_params,
                                                        transform=transform_train,
                                                        joint_transform=transform_joint)

    data_processing_val = processing.KLDiMPProcessing(search_area_factor=settings.search_area_factor,
                                                      output_sz=settings.output_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      mode='sequence',
                                                      proposal_params=proposal_params,
                                                      label_function_params=label_params,
                                                      label_density_params=label_density_params,
                                                      transform=transform_val,
                                                      joint_transform=transform_joint)
    if is_search:
        data_processing_test = processing.KLDiMPProcessing(search_area_factor=settings.search_area_factor,
                                                      output_sz=settings.output_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      mode='sequence',
                                                      proposal_params=proposal_params,
                                                      label_function_params=label_params,
                                                      label_density_params=label_density_params,
                                                      transform=transform_test,
                                                      joint_transform=transform_joint)

    # Train sampler and loader
    if is_search:
        dataset_train = sampler.DiMPSampler([trackingnet_train], [1],
                                            samples_per_epoch=15000, max_gap=200, num_test_frames=3, num_train_frames=3,
                                            processing=data_processing_train)



        # Validation samplers and loaders
        dataset_val = sampler.DiMPSampler([trackingnet_val], [1], samples_per_epoch=15000, max_gap=200,
                                          num_test_frames=3, num_train_frames=3,
                                          processing=data_processing_val)

        loader_val = LTRLoader('val', dataset_val, training=True, batch_size=settings.batch_size,
                               num_workers=settings.num_workers,
                               shuffle=True, drop_last=True, stack_dim=1)

        dataset_test = sampler.DiMPSampler([got10k_test,lasot_test,got10k_val], [1,1,1], samples_per_epoch=5000, max_gap=200,
                                          num_test_frames=3, num_train_frames=3,
                                          processing=data_processing_test)

        loader_test = LTRLoader('test', dataset_test, training=False, batch_size=settings.batch_size,
                               num_workers=settings.num_workers,
                               shuffle=False, drop_last=True, epoch_interval=1, stack_dim=1)

    else:
        dataset_train = sampler.DiMPSampler([got10k_train,lasot_train,coco_train,trackingnet_train], [1,0.25,1,1],
                                        samples_per_epoch=26000, max_gap=200, num_test_frames=3, num_train_frames=3,
                                        processing=data_processing_train)

        dataset_val = sampler.DiMPSampler([got10k_val,trackingnet_val], [1,1], samples_per_epoch=5000, max_gap=200,
                                      num_test_frames=3, num_train_frames=3,
                                      processing=data_processing_val)

        loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size,
                           num_workers=settings.num_workers,
                           shuffle=False, drop_last=True, epoch_interval=5, stack_dim=1)

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=1)



    # Create network and actor
    net = dimpnet.klcedimpnet50(filter_size=settings.target_filter_sz, backbone_pretrained=True, optim_iter=5,
                            clf_feat_norm=True, clf_feat_blocks=0, final_conv=True, out_feature_dim=1024,
                                optim_init_step=1.0, optim_init_reg=0.05, optim_min_reg=0.05,
                                gauss_sigma=output_sigma * settings.feature_sz, alpha_eps=0.05, normalize_label=True, init_initializer='zero', search=is_search)


    if is_search:
        model_dict = net.state_dict()
        sett = ws_settings.Settings()
        model_path = sett.env.pretrained_tracker_dir
        pretrained_dict = {}  # Dictionary to gather pre-trained layers
        pretrained_net = torch.load(model_path)
        counter_main = 0
        counter_nas = 0
        for k, v in pretrained_net.get('net').items():
            if k in model_dict and k.find('classifier') == -1:
                pretrained_dict.update({k: v})
                counter_main += 1

        for k2, v2 in model_dict.items():
            if k2.find('neck') != -1:  # nas_params
                counter_nas += 1

        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        print('**********************Network initilization completed************************')
        print('## Main Items:', counter_main)
        print('## NAS Items: ', counter_nas)
        # assert counter_nas + counter_main == len(model_dict)
        print('*****************************************************************************')

        for n, p in net.named_parameters():
            if ("classifier" in n) or ("neck" in n):
                print(n)
            else:
                p.requires_grad = False


    objective = {'bb_ce': klreg_losses.KLRegression(), 'clf_ce': klreg_losses.KLRegressionGrid()}

    loss_weight = {'bb_ce': 0.0025, 'clf_ce': 0.25, 'clf_ce_init': 0.25, 'clf_ce_iter': 1.0}

    actor = tracking_actors.KLDiMPActor(net=net, objective=objective, loss_weight=loss_weight)

    # Optimizer
    if is_search:
        optimizer = optim.Adam([{'params': actor.net.classifier.parameters(), 'lr': 1e-3},
                                {'params': actor.net.neck.parameters(), 'lr': 1e-3}])

        dummy_optimizer = optim.Adam(
            [{'params': actor.net.neck.parameters(), 'lr': 1e-3}])  # check weight decay! used in search
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)
        dummy_lr_scheduler = optim.lr_scheduler.StepLR(dummy_optimizer, step_size=15, gamma=0.2)


        # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, float(epochs), eta_min=learning_rate_min,
        #     last_epoch=-1 if start_epoch == 0 else start_epoch)

        trainer = LTRTrainer(actor, [loader_train, loader_val], [optimizer, dummy_optimizer], settings,
                             [lr_scheduler, dummy_lr_scheduler], momentum_args=0.9, weight_decay_args=3e-4,
                             arch_learning_rate=3e-4, arch_weight_decay=1e-3, search=is_search, hold_out=loader_test, num_epochs = epochs)
    else:
        optimizer = optim.Adam([{'params': actor.net.classifier.parameters(), 'lr': 1e-3},
                            {'params': actor.net.bb_regressor.parameters(), 'lr': 1e-3},
                            {'params': actor.net.neck.parameters(), 'lr': 1e-3},
                            {'params': actor.net.feature_extractor.parameters(), 'lr': 2e-5}],
                           lr=2e-4)

        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)
        trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, momentum_args=0.9,
                         weight_decay_args=3e-4,
                         arch_learning_rate=3e-4, arch_weight_decay=1e-3, search=is_search)

    trainer.train(epochs, load_latest=True, fail_safe=True)
