import torch
from torch.autograd import Variable
from torch import nn
import numpy as np

# batch Normalization 1D
print("Batch Normalization 1D ...")
m = nn.BatchNorm1d(10)
input = torch.randn(5,10)
output = m(Variable(input))
print("The input shape: {}".format(input.shape))
print("The output shape: {}".format(output.shape))

print("The Batch Normalization parameter is : ")
for name, param in m.named_parameters():
    print(name)
    print(param.data.numpy())

# batch Normalization 2D
print("Batch Normalization 2D ...")
m = nn.BatchNorm2d(10)
input = Variable(torch.randn(3,10,5,5))
output = m(input)
print("The input shape: {}".format(input.shape))
print("The output shape: {}".format(output.shape))

print("The Batch Normalization parameter is : ")
for name, param in m.named_parameters():
    print(name)
    print(param.data.numpy())
