import os
from collections import OrderedDict
from ltr.trainers import BaseTrainer
from ltr.admin.stats import AverageMeter, StatValue
from ltr.admin.tensorboard import TensorboardWriter
import torch
import time

import numpy as np
from visdom import Visdom

from ltr.cnn.architect import Architect
import ltr.admin.settings as ws_settings

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name):
        self.viz = Visdom()#port=8098)
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None,momentum_args=0.9, weight_decay_args=3e-4,arch_learning_rate=3e-4, arch_weight_decay=1e-3,search=False, hold_out=None, num_epochs=50):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler,momentum_args, weight_decay_args, arch_learning_rate, arch_weight_decay, search, hold_out,num_epochs)

        self.search=search
        if search:
            model=actor.net #whole net
            self.architect = Architect(model,momentum_args, weight_decay_args,arch_learning_rate, arch_weight_decay)

        self._set_default_settings()
        # Initialize statistics variables
        if hold_out is not None:
            self.loaders.append(hold_out)
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard
        tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
        self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in self.loaders])

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)

        self.plotter = VisdomLinePlotter(env_name='DimpNas')
    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation.""" #Main train

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        self._init_timing()

        for i, data in enumerate(loader, 1):
            # get inputs
            if self.move_data_to_gpu:
                data = data.to(self.device)

            data['epoch'] = self.epoch
            data['settings'] = self.settings

            # forward pass
            loss, stats = self.actor(data)

            # backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # update statistics
            batch_size = data['train_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            # print statistics
            self._print_stats(i, loader, batch_size)
        self._plot_stats(loader)


    def search_cycle_dataset(self, loaders):
        """Do a cycle of training or validation.""" #Main train
        torch.set_grad_enabled(True)
        self._init_timing()
        sett1 = ws_settings.Settings()

        self.actor.net.neck.p_f = 0.6
        self.actor.net.neck.cells[0]._ops.p = self.actor.net.neck.p_f * (1 - (self.epoch / self.num_epochs))

        dummy_main_lr = self.dummy_lr_scheduler.get_lr()[0]

        for step, (data_train, data_val) in enumerate(zip(loaders[0], loaders[1]), 1):
            # get inputs
            if self.move_data_to_gpu:
                data_train = data_train.to(self.device)
                data_val = data_val.to(self.device)

            data_train['epoch'] = self.epoch
            data_train['settings'] = self.settings

            if self.epoch>10:
                unrolled=True
                self.architect.step([data_train,data_val], dummy_main_lr, self.dummy_optimizer,self.actor, unrolled=unrolled) #main lr & optimizer

            alpha1 = self.actor.net.neck.alphas_normal
            if step % 100 == 0:

                with open( sett1.env.alpha_search_dir + str(self.epoch) + '.txt', 'a') as txt_betas:
                    txt_betas.writelines('Iteration: ' + str(step) + '    Epoch: ' + str(self.epoch))
                    txt_betas.writelines('\n' + 'alpha: ' + str(alpha1))
                    txt_betas.writelines('\n' + '********************************************')
            # forward pass
            loss, stats = self.actor(data_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size = data_train['train_images'].shape[loaders[0].stack_dim]

            self._update_stats(stats, batch_size, loaders[0])

            # print statistics
            self._print_stats(step, loaders[0], batch_size)

        self._plot_stats(loaders[0])

    def train_epoch(self):
        """Do one epoch for each loader."""
        if not self.search:
            for loader in self.loaders:
                if self.epoch % loader.epoch_interval == 0:
                    self.cycle_dataset(loader)
        else:
            self.search_cycle_dataset(self.loaders)
            if self.epoch > 10:
                self.genotype=self.actor.net.neck.genotype()
                print('Geno', self.genotype)
                if self.hold_out is not None:
                    self.cycle_dataset(self.hold_out)

        self._stats_new_epoch()
        self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time
        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)
            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats) and hasattr(val, 'avg'):
                    print_str += '%s: %.5f  ,  ' % (name, val.avg)
            print(print_str[:-5])


    def _plot_stats(self,loader):
         for name, val in self.stats[loader.name].items():
            if (self.settings.print_stats is None or name in self.settings.print_stats) and hasattr(val, 'avg'):
                if (name=='Loss/total'):
                    print('name',name, val.avg,loader.name)
                    self.plotter.plot('loss', loader.name, 'Loss', self.epoch, val.avg)
                #if (loader.name=='Loss/total'):
                #plotter = VisdomLinePlotter(env_name='Plots Total')
                #plotter.plot('loss', name, 'Loss', self.epoch, val.avg)
    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                lr_list = self.lr_scheduler.get_lr()

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.module_name, self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)
