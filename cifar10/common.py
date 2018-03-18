import numpy as np
import torch
from torchvision.datasets import cifar
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

def getDataLoader_Cifar10():
    transform_train = transforms.Compose([
        transforms.Scale(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.Scale(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_set = cifar.CIFAR10('../data/cifar10', train=True, transform=transform_train, download=True)
    test_set = cifar.CIFAR10('../data/cifar10', train=False, transform=transform_test, download=True)
    print(len(train_set))
    train_data = DataLoader(train_set, batch_size=64, shuffle=True)
    test_data = DataLoader(test_set, batch_size=128, shuffle=False)
    return train_data, test_data

class Config:
    def __init__(self, epoch, lr, use_cuda = True):
        self.epoch = epoch
        self.lr =lr
        self.use_cuda = use_cuda

if __name__ == "__main__":
    train_data, test_data = getDataLoader_Cifar10()
    data, label = next(iter(train_data))
    print("Training data len: {}".format(len(train_data)*64))
    print("Test data len: {}".format(len(test_data)*128))
    print(data.shape)