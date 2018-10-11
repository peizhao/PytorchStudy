import torch
from torch.autograd import Variable
from torch import nn
import numpy as np

input = torch.randn(1, 4, 32, 32)
print("General Conv Params: ")
net = nn.Conv2d(4,8,3)
for name, param in net.named_parameters():
    print(name)
    print(param.shape)

print("Group Conv Params: ")
net_G = nn.Conv2d(4,8,3, groups=2)
for name, param in net_G.named_parameters():
    print(name)
    print(param.shape)

output = net(input)
print("Genearal Conv output: ")
print(output.shape)

output = net_G(input)
print("Group Cov output: ")
print(output.shape)


