import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import numpy as np

print("Vector dot with Tensor")
#print(torch.dot(torch.Tensor([2,3]), torch.Tensor([2,1])))
a = torch.Tensor([2,3])
b = torch.Tensor([2,1])
print(torch.dot(a,b))

print("Matrix dot with numpy ...")
a = np.arange(2*3).reshape(3,2)
print("A Matrix")
print(a)
b = np.arange(2).reshape(2,1)
print("B Matrix")
print(b)
c = a.dot(b)
print("dot result")
print(c)

"""
Tensor will use the mm to do the dot ops, torch.dot only employ for the 1D vector
"""
print("Matrix dot with Tensor ...")
a = torch.Tensor([[1,2,3], [1,2,3]]).view(-1,2)
print("A Matrix")
print(a)
b = torch.Tensor([[2,1]]).view(2,-1)
print("B Matrix")
print(b)
c = torch.mm(a,b)
# c= torch.dot(a,b) # this will cause errors
print("dot result")
print(c)
