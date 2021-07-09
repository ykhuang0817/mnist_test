#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ===============================================================
# Copyright (C) 2018 HuangYk.
# Licensed under The MIT Lincese.
#
# Filename     : mnist_engine.py
# Author       : HuangYK
# Last Modified: 2018-07-26 10:27
# Description  :
# ===============================================================


from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchsoa import TorchSoaEngine, ClassifyMeter, SoaRecorder
from torchvision import datasets, transforms


class LeNet(nn.Module):
    '''LeNet-5 networks

    Net structure contains conv1--conv2--dropout--fc1--fc2--dropout--softmax,
    conv is consist of conv2d--max_pool2d--relu, fc is consist of fc--relu,
    classify function is log_softmax, correspond to loss function

    Attribute:
    ----------
    name: Net name for save net model params
    '''
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2_dropout(self.conv2(x))
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def loss_func(self):
        return F.nll_loss

    @property
    def name(self):
        return 'LeNet-5'


class MnistEngine(TorchSoaEngine):
    '''MNIST training process

    This is a training process for MNIST.
    '''
    def __init__(self, *args, **kargs):
        super(MnistEngine, self).__init__(*args, **kargs)

    def get_iterator(self, train):
        mnist_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081,))
            ])
        mnist_dataset = datasets.MNIST('./data/', train=train,
                                       download=True,
                                       transform=mnist_transform)
        return torch.utils.data.DataLoader(
                mnist_dataset,
                batch_size=self._batch_size,
                shuffle=True, num_workers=self._num_workers, pin_memory=True
                )


if __name__ == '__main__':
    batch_size = 256
    num_workers = 10
    num_classes = 10
    max_epoch = 10

    args_momentum = 0.5
    args_lr = 0.01

    n_gpu = 2

    assert torch.cuda.is_available()

    torch.manual_seed(5)

    model = LeNet()

    if n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(n_gpu)))

    if n_gpu > 0:
        model.cuda()

    optimizer = optim.SGD(model.parameters(),
                          lr=args_lr, momentum=args_momentum)

    mnist_params = {'model': model, 'optimizer': optimizer,
                    'maxepoch': max_epoch, 'loss_func': F.nll_loss,
                    'batch_size': batch_size, 'num_workers': num_workers,
                    'net_name': 'LeNet-5'}

    mnist_engine = MnistEngine(**mnist_params)
    mnist_meters = ClassifyMeter(num_classes)

    mnist_epoch_recorder = SoaRecorder(record_step='epoch')
    mnist_epoch_recorder.add_item(kind='confusion', num_classes=num_classes)

    mnist_engine.meters = mnist_meters
    mnist_engine.epoch_rec = mnist_epoch_recorder

    mnist_engine.train()
