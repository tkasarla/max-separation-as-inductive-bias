#
# NeurIPS 2022 Paper ID 1648 code sumbission
# NeurIPS 2022 Paper Title: Maximum Class Separation as Inductive Bias in One Matrix
#
# Helper functions.
#

import os
import sys
import re
import datetime

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.alexnet import alexnet
from models.resnet import resnet18, resnet34, resnet50, resnet101

#
#
#
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#
# Load specified model.
#
def get_model(network, dims):
    if network == "alexnet":
        model = alexnet(dims)
    if network == "resnet34":
        model = resnet34(dims)
    elif network == "resnet50":
        model = resnet50(dims)
    elif network == "resnet101":
        model = resnet101(dims)
    return model.cuda()

#
# Load CIFAR100.
#
def get_cifar100(network,batch_size):
    # Mean and std pixel values.
    cmean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    cstd  = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    # Transforms.

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(cmean, cstd)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cmean, cstd)
    ])

    # Get train loader.
    train_data = torchvision.datasets.CIFAR100(root="../../datasets/", train=True, transform=transform_train)
    train_loader = DataLoader(train_data, shuffle=True, num_workers=32, batch_size=batch_size)

    # Get test loader.
    test_data = torchvision.datasets.CIFAR100(root="../../datasets/", train=False, transform=transform_test)
    test_loader = DataLoader(test_data, shuffle=False, num_workers=32, batch_size=batch_size)

    return train_loader, test_loader

#
# Load CIFAR10.
#
def get_cifar10(network,batch_size):
    # Mean and std pixel values.
    cmean = (0.4914, 0.4822, 0.4465)
    cstd  = (0.2023, 0.1994, 0.2010)

    # Transforms.

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(cmean, cstd)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cmean, cstd)
    ])

    # Get train loader.
    train_data = torchvision.datasets.CIFAR10(root='/path/to/cifar10', train=True, transform=transform_train)#, download=True)
    train_loader = DataLoader(train_data, shuffle=True, num_workers=32, batch_size=batch_size)

    # Get test loader.
    test_data = torchvision.datasets.CIFAR10(root='/path/to/cifar100', train=False, transform=transform_test)#, download=True)
    test_loader = DataLoader(test_data, shuffle=False, num_workers=32, batch_size=batch_size)

    return train_loader, test_loader

#
# Load MNIST.
#
def get_mnist(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))])
    train_data = torchvision.datasets.MNIST("/path/to/mnist", train=True, transform=transform)
    train_loader = DataLoader(train_data, shuffle=True, num_workers=32, batch_size=batch_size)
    test_data = torchvision.datasets.MNIST("/path/to/mnist", train=False, transform=transform)
    test_loader = DataLoader(test_data, shuffle=False, num_workers=32, batch_size=batch_size)
    return train_loader, test_loader


# Legacy: warm-up training lr scheduler.
#
class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
