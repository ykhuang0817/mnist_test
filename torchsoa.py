#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ===============================================================
# Copyright (C) 2018 HuangYk.
# Licensed under The MIT Lincese.
#
# Filename     : torchsoa.py
# Author       : HuangYK
# Last Modified: 2018-08-12 14:15
# Description  :
#
# ===============================================================

import os
import copy
import torch
import torchnet as tnt
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger
import time
import numpy as np
import pandas as pd

from tqdm import tqdm  # progress bar using in python shell
from pandas import DataFrame
from collections import defaultdict


class TorchSoaEngine(object):
    '''A architecture of training process

    Inherit TorchSoaEngine to build a neural network training processor for
    specific dataset, and override abstract method get_iterator to provide a
    batch sample iterator from dataset.

    Attribute:
    ----------
    meters: Caculate loss, class accuracy, class confusion performance of
            neural networks
    model: Neural networks model at gpu device
    parameters: Total number of parameters in model

    Example:
    --------
    >> kw={'model':neural_network_instance,
           'optimizer':optimizer_instance,
           'loss_func':loss_function
           'maxepoch':max_epoch, 'batch_size':batch_size,
           'num_workers':num_workers}
    >> net_engine = TorchSoaEngine(**kw)
    >> net_engine.meters = ClassifyMeter(num_classes)
    >> net_engine.train()
    '''
    def __init__(self, model, optimizer, loss_func, maxepoch, batch_size,
                 num_workers, net_name, **kws):
        '''Init with training parameters, add hooks in torchnet

        Training hooks function sequence is:
            --> hook['on_start']
              --> maxepoch iteration(
                --> hook['on_start_epoch']
                --> batch data iteration(
                  --> state['sample'] --> hook['on_sample']
                  --> state['optimizer'].zero
                  --> forward: state['network'](state['sample'])
                  --> state['output'], state['loss']
                  --> hook['on_forward'] with state['output'] and state['loss']
                  --> state['output'].zero, state['loss'].zero
                  --> backprop: state['optimizer'] with loss
                  --> hook['on_upadte']
                  --> state['t'].add
                ) # one epoch
                --> state['epoch'].add
                --> hook['on_end_epoch']
              ) # one training
            --> hook['on_end']

        Args:
        -----
        model: torch.nn.Module A nerual networks inherit nn.Module
        optimizer: torch.optim Optim method for training
        loss_func： torch.nn.functional, Loss function for nerual networks
        max_epoch: int, Epoch number for training process
        batch_size: int, Sample batch in a iteration
        num_workers: int, Number of processors for get sample
        net_name: str,

        Return:
        -------
        A normalized torch net training architecture
        '''
        self._model = model
        self._optimizer = optimizer
        self._max_epoch = maxepoch
        self._loss_func = loss_func
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._net_name = net_name
        self._epoch_meters = None
        self._epoch_recorder = None

        self._engine = Engine()
        self._engine.hooks['on_sample'] = self._on_sample
        self._engine.hooks['on_forward'] = self._on_forward
        self._engine.hooks['on_start_epoch'] = self._on_start_epoch
        self._engine.hooks['on_end_epoch'] = self._on_end_epoch
        self._engine.hooks['on_end'] = self._on_end

    @property
    def meters(self):
        return self._epoch_meters

    @meters.setter
    def meters(self, meters):
        self._epoch_meters = meters

    @property
    def epoch_rec(self):
        return self._epoch_recorder

    @epoch_rec.setter
    def epoch_rec(self, epoch_rec):
        self._epoch_recorder = epoch_rec

    @property
    def model(self):
        return self._model

    @property
    def parameters(self):
        return sum(param.numel for param in self._model.parameters())

    def _on_start(self):
        pass

    def _on_sample(self, state):
        '''Attach train(True) or test(False) label to samples

        Args:
        -----
        state: dict, a state dict in torchnet, state['sample'] will provide
               a list contain data, target
        '''
        state['sample'].append(state['train'])

    def _on_start_epoch(self, state):
        self._epoch_meters.reset_meters()
        state['iterator'] = tqdm(state['iterator'])

    def _on_forward(self, state):
        '''Process forward output, loss before reset

        Args:
        -----
        state: dict, provide output tensor and loss in state['output'],
               state['loss']
        '''
        self._epoch_meters.add_output_to_meters(state)

    def _on_update(self):
        pass

    def _on_end_epoch(self, state):
        epoch_meters = self._epoch_meters
        epoch_recorder = self._epoch_recorder

        epoch_meters.print_meters(epoch=state['epoch'], train=True)
        epoch_meters.send_meters(epoch=state['epoch'], train=True)
        epoch_recorder.record(
            index=state['epoch'], train=True,
            loss=epoch_meters.loss, accuracy=epoch_meters.accuracy,
            diag=epoch_meters.get_confusion_diag()[0],
            num=epoch_meters.get_confusion_diag()[1]
        )

        epoch_meters.reset_meters()

        self.test()
        epoch_meters.print_meters(epoch=state['epoch'], train=False)
        epoch_meters.send_meters(epoch=state['epoch'], train=False)
        epoch_recorder.record(
            index=state['epoch'], train=False,
            loss=epoch_meters.loss, accuracy=epoch_meters.accuracy,
            diag=epoch_meters.get_confusion_diag()[0],
            num=epoch_meters.get_confusion_diag()[1],
            conf=epoch_meters.get_confusion_matrix()
        )

        torch.save(self._model.state_dict(),
                   'epochs/{:s}_epoch_{:d}.pt'.format(
                       self._net_name, state['epoch']))

    def _processor(self, sample):
        data, target, train = sample
        data = data.cuda()
        target = target.cuda()

        if train:
            self._model.train()
        else:
            self._model.eval()

        output = self._model(data)
        loss = self._loss_func(output, target)

        return loss, output

    def _on_end(self, state):
        '''Save training record
        '''
        csv_folder = './logs'
        if state['train']:
            csv_file = '_'.join(
                [self._net_name, 'epoch', str(self._max_epoch)]
            )
        else:
            csv_file = '_'.join([self._net_name, 'epoch', 'tmp'])

        csv_file = os.path.join(csv_folder, csv_file)
        self._epoch_recorder.save_csv(csv_file, state['train'])

    def get_iterator(self, train):
        raise NotImplementedError(
            'get_iterator not implemented for TorchSoaEngine, which is an \
            abstract class')

    def train(self):
        self._engine.train(self._processor, self.get_iterator(True),
                           maxepoch=self._max_epoch, optimizer=self._optimizer)

    def test(self):
        self._engine.test(self._processor, self.get_iterator(False))


class ClassifyMeter(object):
    '''Classify task performance evaluation with loss curve, accuracy curve,
    confusion matrix

    This class provides loss, accuracy, confusion

    Attribute:
    ----------
    vis: ClassifyVisdom instance for plot loss, accuracy, confusion in
         visdom server in real time during training
    loss: float, average loss
    accuracy: float, average accuracy of total samples
    confusion: [k x k] np.array, class confusion matrix
    '''
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.loss_meter = tnt.meter.AverageValueMeter()
        self.acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
        self.confusion_meter = tnt.meter.ConfusionMeter(
            num_classes, normalized=True)

        self._meters = [self.loss_meter, self.acc_meter, self.confusion_meter]
        self._loggers = ClassifyVisdom(num_classes)

    @property
    def vis(self):
        '''
        Return a meter list contain loss, acc, confusion
        '''
        return self._loggers

    @property
    def loss(self):
        '''
        Return average loss
        '''
        return self.loss_meter.value()[0]

    @property
    def accuracy(self):
        '''
        Return average class accuracy
        '''
        return self.acc_meter.value()[0]

    @property
    def confusion(self):
        '''
        Return confusion matrix of [num_classes x num_classes]
        '''
        self.confusion_meter.normalized = True
        return self.confusion_meter.value()

    def get_confusion_diag(self):
        confusion = self.confusion_meter.conf
        return np.diag(confusion), confusion.sum(1).clip(min=1e-12)

    def get_confusion_matrix(self):
        return self.confusion_meter.conf

    def reset_meters(self):
        for meter in self._meters:
            meter.reset()

    def print_meters(self, epoch=None, train=None):
        process = 'Training' if train else 'Test'
        print('[Epoch {:d}] {:s} Loss: {:.4f} (Accuracy: {:.2f}%)'.
              format(epoch, process, self.loss, self.accuracy))

    def send_meters(self, epoch=None, train=None):
        self._loggers.log(epoch, self.loss, self.accuracy,
                          self.confusion, train)

    def add_output_to_meters(self, state):
        '''Add output, target to meters(loss, acc, confusion) per batch iter

        Args:
        -----
        state: dict, provide loss, output, target
        '''
        self.loss_meter.add(state['loss'].data.item())
        self.acc_meter.add(state['output'].data, state['sample'][1])
        self.confusion_meter.add(state['output'].data, state['sample'][1])


class ClassifyVisdom(object):
    '''Visdom logger for classify task, contain loss curve, accuracy curve and
    confusion matrix, plot in visdom server
    '''
    def __init__(self, num_classes):
        self._loss_logger = LossVisdom()
        self._acc_logger = AccuracyVisdom()
        self._confusion_logger = ConfusionVisdom(num_classes)

    def log(self, epoch, loss, accuracy, confusion, train=None):
        self._loss_logger.log(epoch, loss, train)
        self._acc_logger.log(epoch, accuracy, train)
        self._confusion_logger.log(confusion, train)


class LossVisdom(object):
    '''Plot train and test loss curve together in a VisdomPlotLogger
    '''
    def __init__(self):
        self._loss = VisdomPlotLogger('line', opts={
            'title': 'Loss Curve'
        })
        check_visdom_server(self._loss.viz)

    def log(self, epoch, loss, train=None):
        assert train is not None,\
            'train should be True or False, not {}'.format(train)
        name = 'train' if train else 'test'
        self._loss.log(epoch, loss, name=name)


class AccuracyVisdom(object):
    '''Plot train and test accuracy curve together in a VisdomPlotLogger
    '''
    def __init__(self):
        self._acc = VisdomPlotLogger('line', opts={
            'title': 'Accuracy Curve'
        })
        check_visdom_server(self._acc.viz)

    def log(self, epoch, accuracy, train=None):
        assert train is not None,\
            'train should be True or False, not {}'.format(train)
        name = 'train' if train else 'test'
        self._acc.log(epoch, accuracy, name=name)


class ConfusionVisdom(object):
    '''Plot test confusion matrix in a VisdomLogger
    '''
    def __init__(self, num_classes):
        self._confusion = VisdomLogger('heatmap', opts={
            'title': 'Confusion Matrix',
            'columnnames': list(range(num_classes)),
            'rownames': list(range(num_classes))
        })
        check_visdom_server(self._confusion.viz)

    def log(self, confusion, train=None):
        assert train is not None,\
            'train should be True or False, not {}'.format(train)
        if train:
            pass
        else:
            self._confusion.log(confusion)


class SoaRecorder(object):
    '''Record loss and accuracy of a training process as csv

    '''
    items = ['loss-acc']

    def __init__(self, record_step):
        assert self.check_default_save_folder(), 'Save folder created failed'
        self.record_step = record_step
        self._recs = defaultdict(lambda: 'N/A')
        self._recs['loss-acc'] = LossAccRecorder(record_step)

    def check_default_save_folder(self, path='./logs'):
        if os.path.exists(path):
            return True
        else:
            os.makedirs(path)
            self.check_default_save_folder(path)

    def add_item(self, kind, num_classes):
        assert kind in ['confusion'], 'Record type not support'
        if kind == 'confusion':
            self.items.append(kind)
            self._recs[kind] = ConfusionRecorder(
                self.record_step, num_classes
            )

    def get_record(self):
        '''
        Return: A dict of DataFrame, which index in items
        '''
        return self._recs

    def record(self, index, train, loss=np.nan, accuracy=np.nan,
               diag=np.nan, num=np.nan, conf=None):
        '''Add loss, accuracy to DataFrame

        Args:
        -----
        index: int, epoch or batch iteration number
        loss: float, loss of net forward process in this index
        accuracy: float, average accuracy among classes in this index
        train: boolean, if this index is a training process
        '''
        kws = {'index': index, 'train': train, 'loss': loss, 'conf': conf,
               'accuracy': accuracy, 'diag': diag, 'num': num}
        for kind in self.items:
            self._recs[kind].record(**kws)

    def save_csv(self, path, train=None):
        for item in self.items:
            if not self._recs[item] == 'N/A':
                self._recs[item].save_csv(path, train)
            else:
                print('{} not used'.format(item))


class LossAccRecorder(object):
    '''
    '''
    def __init__(self, record_step):
        self.record_step = record_step
        self._df = DataFrame(
            columns=[['loss', 'loss', 'accuracy', 'accuracy'],
                     ['train', 'test', 'train', 'test']]
            )

        self._df.index.name = record_step

    def record(self, index, train, loss, accuracy, **kws):
        c_level1 = 'train' if train else 'test'
        self._df.loc[index, ('loss', (c_level1))] = loss
        self._df.loc[index, ('accuracy', (c_level1))] = accuracy

    def save_csv(self, path, train):
        self._df.to_csv('{0:s}_loss-acc.csv'.format(path))


class ConfusionRecorder(object):
    '''
    '''
    items = ['diag_train', 'diag_test', 'num_train', 'num_test']

    def __init__(self, record_step, num_classes):
        self.record_step = record_step
        self._dfs = defaultdict(lambda: 'N/A')
        self._confs = []
        self._confs_keys = []
        for k in self.items:
            self._dfs[k] = DataFrame(columns=np.arange(num_classes))

    def record(self, index, train, diag, num, conf=None, **kws):
        diag_key = 'diag_train' if train else 'diag_test'
        num_key = 'num_train' if train else 'num_test'
        self._dfs[diag_key].loc[index] = diag
        self._dfs[num_key].loc[index] = num
        if conf is not None and not train:
            conf_df = DataFrame(conf)
            conf_df.to_csv(
                './logs/{0:s}_{1:d}_test_confusion.csv'.format(
                    self.record_step, index)
            )
            self._confs.append(copy.deepcopy(conf_df))
            self._confs_keys.append('epoch_{:d}'.format(index))

    def save_csv(self, path, train):
        df = pd.concat(
            [self._dfs['diag_train'], self._dfs['diag_test'],
             self._dfs['num_train'], self._dfs['num_test']],
            axis=1, keys=self.items
        )
        df.index.name = self.record_step
        df.to_csv('{:s}_diag.csv'.format(path))
        if len(self._confs) > 0:
            conf_concat_df = pd.concat(
                self._confs, axis=1, keys=self._confs_keys
            )
            conf_concat_df.index.name = 'Target'
            conf_concat_df.to_csv('{:s}_confusion.csv'.format(path))


def check_visdom_server(vis):
    '''check if visdom server start up

    Args:
    -----
    vis: visdom.Visdom isinstance

    Return:
    -------
    Throw a assert exception if visdom server not work,
    return none if visdom server is running
    '''
    startup_sec = 1
    while not vis.check_connection() and startup_sec > 0:
        time.sleep(0.1)
        startup_sec -= 0.1
    assert vis.check_connection(), 'No visdom server found, \
use python -m visdom.server to start a visdom server'
