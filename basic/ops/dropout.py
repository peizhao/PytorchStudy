import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import numpy as np

m = nn.Dropout(p=0.3)
input = Variable(torch.randn(1,10))
print("Input value is: {}".format(input))
output = m(input)
print("Output value is: {}".format(output))