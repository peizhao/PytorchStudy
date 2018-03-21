import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import numpy as np

input = Variable(torch.randn(1,5))
print(input)
output = F.softmax(input,1)
print(output)

input= [1,2,3,4,2,2,1]
data = torch.from_numpy(np.asarray(input, dtype=np.float32))
output = F.softmax(Variable(data),0)
print(output)