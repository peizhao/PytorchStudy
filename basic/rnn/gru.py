import torch
from torch.autograd import Variable
from torch import nn
import numpy as np

input_size = 5
hidden_size = 10
layer_size = 3
net = nn.GRU(input_size, hidden_size, layer_size)

print("The weight and bias matrix is as bleow:")
print("layer 0: ...")
print(net.weight_ih_l0.shape)
print(net.bias_ih_l0.shape)
print(net.weight_hh_l0.shape)
print(net.bias_hh_l0.shape)
print("layer 1: ...")
print(net.weight_ih_l1.shape)
print(net.bias_ih_l1.shape)
print(net.weight_hh_l1.shape)
print(net.bias_hh_l1.shape)
print("layer 2: ...")
print(net.weight_ih_l2.shape)
print(net.bias_ih_l2.shape)
print(net.weight_hh_l2.shape)
print(net.bias_hh_l2.shape)