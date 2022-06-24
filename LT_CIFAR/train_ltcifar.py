#
# NeurIPS 2022 Paper ID 1648 code sumbission
# NeurIPS 2022 Paper Title: Maximum Class Separation as Inductive Bias in One Matrix
#
# Train and test on CIFAR-10 / CIFAR-100 for:
# - softmax cross-entropy
# - Softmax with matrix with maximum class separation
#

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from lt_cifar import CIFAR10_LT, CIFAR100_LT

import utils

#
# Train for one epoch.
#
def train(model, train_loader, approach, loss_function, prototypes, epoch):
    model.train()
    total = 0
    # Iterate over all samples.
    for batch_index, (images, labels) in enumerate(train_loader):
        total += len(images)
        print("TRAIN %d [%d/%d]" %(epoch, total, len(train_loader.dataset)), end="\r")
        # Images and labels to GPU.
        images = images.cuda()
        labels = labels.cuda()

        # Forward propagation.
        optimizer.zero_grad()
        outputs = model(images)
        # Compute loss.
        if args.approach == "sce":
            loss = loss_function(outputs, labels)
        elif args.approach == "msce":
            outputs = torch.mm(outputs, prototypes)
            loss = loss_function(outputs, labels)
        # Backward propagation and update.
        loss.backward()
        optimizer.step()

#
# Test for one epoch.
#
def test(model, test_loader, approach, loss_function, prototypes):
    model.eval()
    correct = 0.0
    # Go over all test samples.
    with torch.no_grad():
        for (images, labels) in test_loader:
            # Images and labels to GPU.
            images = images.cuda()
            labels = labels.cuda()

            # Forward propagation.
            outputs = model(images)

            # Prediction.
            if approach == "msce":
                outputs = torch.mm(outputs, prototypes)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()

    return correct.float() / len(test_loader.dataset)


if __name__ == '__main__':
    # Parse user arguments.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", dest="network", default="resnet34", type=str,help='alexnet or resnet34') #resnet34, alexnet
    parser.add_argument("-d", dest="dataset", default="cifar100", type=str,help='cifar10 or cifar100') #cifar10, cifar100
    parser.add_argument("-a", dest="approach", default="sce", type=str,help='loss with only softmax(sce) or with maximum separation(msce)') #sce, msce
    parser.add_argument("-p", dest="prototypes", default="", type=str,'location of the maximum separation matrix, needed if loss is msce')
    parser.add_argument("-r", dest="radius", default=1.0, type=float,help='radius of the prototypes, use 1 for alexnet and 0.1 for resnet')
    parser.add_argument("-b", dest="batch_size", default=512, type=int,help='batch size of the training')
    parser.add_argument("-l", dest="learning_rate", default=0.1, type=float,help='learning rate of the algorithm')
    parser.add_argument("-e", dest="epochs", default=200, type=int, help='number of epochs to train the network')
    parser.add_argument("-i", dest="imbalance_factor", default=0.01, type=float,help='imbalance factor of cifar dataset. common values are 0.2,0.1,0.02,0.01')
    parser.add_argument("-f", dest="resfile", default="", type=str,help='location of logfile to be saved')
    parser.add_argument("-sd", dest = "seed", default=123, type=int, help='seed for initializing training.')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Get data.
    if args.dataset == "cifar100":
        print("loading cifar-100 dataset with imabalance factor "+ str(args.imbalance_factor)+" ...")
        dataset = CIFAR100_LT(imb_factor=args.imbalance_factor, batch_size=args.batch_size)
    else:
        print("loading cifar-10 dataset with imabalance factor "+ str(args.imbalance_factor)+" ...")
        dataset = CIFAR10_LT(imb_factor=args.imbalance_factor, batch_size=args.batch_size)
    train_loader = dataset.train_instance
    test_loader  = dataset.eval

    # Get loss function and optional prototypes.
    if args.approach == "sce":
        loss_function = nn.CrossEntropyLoss()
        prototypes = None
        test_prototypes = None
        dims = 100 if args.dataset == "cifar100" else 10
    elif args.approach == "msce":
        loss_function = nn.CrossEntropyLoss()
        prototypes  = torch.from_numpy(np.load(args.prototypes)).float()
        prototypes *= args.radius
        dims = prototypes.shape[1]
        prototypes = prototypes.t().cuda()
        test_prototypes = prototypes

    # Get network.
    model = utils_an.get_model(args.network, dims)

    # Get optimizer and optional scheduler.
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    if args.network == "alexnet":
        train_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0)

    # Open logger.
    do_log = False
    if args.resfile != "":
        logger = open(args.resfile, "a")
        do_log = True

    #creating a checkpoints folder
    if not os.path.exists('ckpts'):
        os.makedirs('ckpts')

    # Perform training and periodic testing.
    for epoch in range(args.epochs):
        # Train
        train(model, train_loader, args.approach, loss_function, prototypes, epoch)

        # Test.
        if epoch % 10 == 0 or epoch == args.epochs -1:
            acc = test(model, test_loader, args.approach, loss_function, test_prototypes)
            print()
            logline = "TEST [%s: e-%d a-%s if-%.3f d-%d-%.3f n-%s l-%.3f] : %.4f" %(args.dataset, epoch, args.approach, args.imbalance_factor, dims, args.radius, args.network, train_scheduler.get_last_lr()[0], acc)
            torch.save(model.state_dict(), 'ckpts/'+args.dataset+'_lt_'+'_if'+str(args.imbalance_factor)+'_'+args.network+'_'+args.approach+'.pt')
            if do_log:
                logger.write(logline + "\n")
            print(logline)

        # Learning rate scheduler update.
        train_scheduler.step()

    if do_log:
        logger.write("\n")
    print()
