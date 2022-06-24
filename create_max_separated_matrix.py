#
# NeurIPS 2022 Paper ID 1648 code sumbission
# NeurIPS 2022 Paper Title: Maximum Class Separation as Inductive Bias in One Matrix
#
# Code to generate the matrix with maximum separation.
# A demo code is in LT_CIFAR folder to evalute the matrix on CIFAR-10 and CIFAR-100
# You can also load it directly in your code by calling the create_prototypes() function; without needing to save as a npy file.
#

import sys
sys.setrecursionlimit(10000) #for nr_prototypes>=1000
import numpy as np
from scipy.spatial.distance import *

def create_noisy_prototypes(nr_prototypes, noise_scale=0):
    prototypes = create_prototypes(nr_prototypes)
    if noise_scale != 0:
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=prototypes.shape)
        prototypes = L2_norm(prototypes + noise)
    distances = cdist(prototypes, prototypes)
    avg_dist = distances[~np.eye(*distances.shape, dtype=bool)].mean()
    return prototypes.astype(np.float32), avg_dist

def create_prototypes(nr_prototypes):
    assert nr_prototypes > 0
    prototypes = V(nr_prototypes - 1).T
    assert prototypes.shape == (nr_prototypes, nr_prototypes - 1)
    assert np.all(np.abs(np.sum(np.power(prototypes, 2), axis=1) - 1) <= 1e-6)
    distances = cdist(prototypes, prototypes)
    assert distances[~np.eye(*distances.shape, dtype=bool)].std() <= 1e-3
    return prototypes.astype(np.float32)

def create_prototypes_random(nr_prototypes):
    prototypes = L2_norm(np.random.uniform(size=(nr_prototypes, nr_prototypes - 1)))
    assert prototypes.shape == (nr_prototypes, nr_prototypes - 1)
    assert np.all(np.abs(np.sum(np.power(prototypes, 2), axis=1) - 1) <= 1e-6)
    return prototypes.astype(np.float32)

def V(order):
    if order == 1:
        return np.array([[1, -1]])
    else:
        col1 = np.zeros((order, 1))
        col1[0] = 1
        row1 = -1 / order * np.ones((1, order))
        return np.concatenate((col1, np.concatenate((row1, np.sqrt(1 - 1 / (order**2)) * V(order - 1)), axis=0)), axis=1)

if __name__ == '__main__':

    nr_classes = 10
    prototypes = create_prototypes(nr_classes)
    np.save("prototypes"+str(nr_classes)+".npy", prototypes)
