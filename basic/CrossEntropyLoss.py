import numpy as np
import torch
from torchvision.datasets import mnist
from torch import nn
from torch.autograd import Variable


loss = nn.CrossEntropyLoss()
input = Variable(torch.randn(3,5), requires_grad=True)
print(input)
target =Variable(torch.LongTensor(3).random_(5))
print target
output = loss(input, target)
print output

"""
The Matrix CrossEntropyLoss is equal the avg loss of every ROW
"""
label = torch.randn(3,5)
label0 = label[0,:].resize_(1,5)
label1 = label[1,:].resize_(1,5)
label2 = label[2,:].resize_(1,5)
input = Variable(label, requires_grad=True)
target = torch.LongTensor(3).random_(5)
target0 = torch.LongTensor([target[0]])
target1 = torch.LongTensor([target[1]])
target2 = torch.LongTensor([target[2]])
all_loss = loss(input, Variable(target))
print("Total loss is: {}".format(all_loss))
loss0 = loss(Variable(label0), Variable(target0))
loss1 = loss(Variable(label1), Variable(target1))
loss2 = loss(Variable(label2), Variable(target2))
print("Loss 0 is : {}".format(loss0))
print("Loss 1 is : {}".format(loss1))
print("Loss 1 is : {}".format(loss2))
print("Avg loss is : {}".format((loss0+loss1+loss2)/3))