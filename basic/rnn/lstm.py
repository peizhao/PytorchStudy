import torch
from torch.autograd import Variable
from torch import nn
import numpy as np

input_size = 5
hidden_size = 10
layer_size = 3
net = nn.LSTM(input_size, hidden_size, layer_size)

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


print("The input and output for LSTM ...")
print("input shape is (3,1,50)")
x = Variable(torch.randn(3,1,50)) # input
net = nn.LSTM(50,1)
lstm_seq_out, _ = net(x)
print("lstm_seq_out shape is {}".format(lstm_seq_out.shape))
