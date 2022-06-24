#
# NeurIPS 2022 Paper ID 1648 code sumbission
# NeurIPS 2022 Paper Title: Maximum Class Separation as Inductive Bias in One Matrix
#
# Evaluate on CIFAR-10 / CIFAR-100 for:
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

import utils
#
def test(model, test_loader, approach, loss_function, prototypes):
    correct = 0.0
    model.eval()
    nb_classes = 10
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    # Go over all test samples.
    with torch.no_grad():
        for (images, labels) in test_loader:
            # Images and labels to GPU.
            images = images.cuda()
            labels = labels.cuda()

            # Forward propagation.
            outputs = model(images)
            # print(outputs.shape)

            # Prediction.
            if approach == "msce":
                outputs = torch.mm(outputs, prototypes)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()
            for t, p in zip(labels.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
            # print(correct)

        return correct.float() / len(test_loader.dataset),confusion_matrix


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", dest="network", default="resnet34", type=str)
    parser.add_argument("-d", dest="dataset", default="cifar10", type=str)
    parser.add_argument("-a", dest="approach", default="sce", type=str) # options : sce, hpn, hsce
    parser.add_argument("-p", dest="prototypes", default="", type=str) #location of prototypes
    parser.add_argument("-r", dest="radius", default=1.0, type=float) #can it be changed (?)
    parser.add_argument("-g", dest="radius_growrate", default=1.0, type=float) # increases or decreases radius (?)
    parser.add_argument("-b", dest="batch_size", default=512, type=int) # fixed or can be changed (?)

    args = parser.parse_args()

    # Get data.
    if args.dataset == "cifar100":
        print("loading cifar-100 dataset...")
        train_loader, test_loader = utils.get_cifar100(args.network,args.batch_size)
    else:
        print("loading cifar-10 dataset...")
        train_loader, test_loader = utils.get_cifar10(args.network,args.batch_size)

    # Defining loss function and loading optional prototypes.
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

    # print(dims)
    model = utils.get_model(args.network, dims)
    model.load_state_dict(torch.load('ckpts/cifar10_lt_if0.02_resnet34_msce.pt'))


    acc,confusion_matrix = test(model, test_loader, args.approach, loss_function, test_prototypes)
    print("Overall Accuracy:", acc)
    per_class_acc = confusion_matrix.diag()/confusion_matrix.sum(1)
    print("Per Class Accuracy:", per_class_acc)
