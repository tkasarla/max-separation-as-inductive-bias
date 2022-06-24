
# Introduction

This repository is the official implementation of CIFAR evaluation for NeurIPS 2022 submission : **Maximum Class Separation as Inductive Bias in One Matrix**


# Requirements

All the codes are run in a miniconda3 environment with python version 3.7

To install requirements (ignore this if you already ran it in the main folder):

```setup
pip install -r requirements.txt
```

Download CIFAR-10 and CIFAR-100 to your desired location and change that dataset root location in `utils.py` and `lt_cifar.py` to load the datasets correctly.

# Training

To train the model(s) in the paper, run the following commands.

### CIFAR dataset -- balanced

To train balanced CIFAR-10 dataset with only Softmax Loss, run:

```train
python train_cifar.py -n resnet34 -d cifar10 -a sce -p None -l 0.1 -b 512 -e 200
```
If network is alexnet change -e to 90

To train the balanced CIFAR-10 dataset with maximum class separation matrix generated from `create_max_separated_matrix.py`, run:

```train
python train_cifar.py -n resnet34 -d cifar10 -a msce -p /path/to/prototypes-10.npy -r 0.1 -b 512 -e 200
```
If network is alexnet, change -e to 90 and -r to 1.0

To train balanced CIFAR-100 datasets, only change the `-d` to `cifar100` and the optional `-p` parameter to `path/to/prototypes100.npy` in training script

### CIFAR dataset -- imbalanced

To train imbalance CIFAR-10 dataset with only Softmax Loss (values for i are 0.2,0.1,0.02,0.01), run:

```train
python train_cifar.py -n resnet34 -d cifar10 -a sce -p None -l 0.1 -b 512 -i 0.1 -e 200
```
If network is alexnet change -e to 90

To train the imbalanced CIFAR-10 dataset with maximum class separation matrix generated from `create_max_separated_matrix.py` (values for i are 0.2,0.1,0.02,0.01), run:

```train
python train_cifar.py -n resnet34 -d cifar10 -a msce -p /path/to/prototypes-10.npy -r 0.1 -b 512 -i 0.1 -e 200
```
If network is alexnet, change -e to 90 and -r to 1.0

To train imbalanced CIFAR-100 datasets, only change the `-d` to `cifar100` and the optional `-p` parameter to `path/to/prototypes100.npy` in training script

### A word about imbalance factor
The imbalance dataset for CIFAR was introduced in [LDAM-DRW paper](https://arxiv.org/pdf/1906.07413.pdf). We use the same settings and use the code from [MiSLAS](https://github.com/dvlab-research/MiSLAS) to generate the imbalanced datasets.

the imbalance factor -i in the training scripts indicates the ratio between least number of samples and highest number of samples in the CIFAR datasets.  


## Evaluation

Remember to change the location of saved model in the code.

To evaluate the model trained on only Softmax Loss, run:

```eval
python eval_model.py -d cifar10
```

To evaluate the model trained on the matrix with maximum class separation, run:

```eval
python eval_model.py -d cifar10 -a msce -p /path/to/prototypes-10.npy

To evaluate on CIFAR-100 dataset, only change the `-d` to `cifar100` and the optional `-p` parameter to `path/to/prototypes100.npy` in training script.
