import torch
from torch.autograd import Variable
from torch import nn
import numpy as np

print("The parameters of the ConvTranspose2d ...")
print("The general parameter : ")
net = nn.ConvTranspose2d(3,8,3)
for name, param in net.named_parameters():
    print(name)
    print(param.shape)

print("The parameter without bias : ")
net = nn.ConvTranspose2d(3,8,3,bias=False)
for name, param in net.named_parameters():
    print(name)
    print(param.shape)

print("The output for one pixel: ")
input = torch.randn(1,1,1,1)
net = nn.ConvTranspose2d(1, 8, 4, 1, 0, bias=False)
output = net(Variable(input))
print(output.shape)

print("Basic output of the ConvTranspose2d ....")
input = torch.randn(1,16,12,12)
m = nn.ConvTranspose2d(16,33,3,stride=2)
output = m(Variable(input))
print("input shape is {}".format(input.shape))
print("output shape is {}".format(output.shape))

print("conv vs transpose conv ....")
input = torch.randn(1,16,12,12)
conv_down = nn.Conv2d(16,16,3,stride=2, padding=1)
trans_conv_up = nn.ConvTranspose2d(16,16,3,stride=2, padding=1)
print("original input size: {}".format(input.shape))
down_result = conv_down(Variable(input))
print("after conv_down size:  {}".format(down_result.shape))
up_result = trans_conv_up(down_result)
print("after trans_conv_up size: {}".format(up_result.shape))
up_result = trans_conv_up(down_result, output_size = input.size())
print("after trans_conv_up with assign size, the size: {}".format(up_result.shape))
