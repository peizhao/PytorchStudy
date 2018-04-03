import torch
from torch.autograd import Variable
from torch import nn
import numpy as np

x = torch.randn(3)
x = Variable(x,requires_grad = True)
target =torch.randn(2)
target = Variable(target)

f = nn.Linear(3,2)

#lossFunction =  nn.CrossEntropyLoss()
lossFunction = nn.MSELoss()
optimizer = torch.optim.SGD(f.parameters(), 1)
print("Input Value is: ")
print(x)
print("Target Value is: ")
print(target)
print("The init parameter is: ")
for name, param in f.named_parameters():
    print(name + " .... ")
    print(param.data.numpy())

for i in range(3):
    print("Iter %d ......." % i)
    y = f(x)
    optimizer.zero_grad()
    loss = lossFunction(y,target)
    loss.backward()
    for name, param in f.named_parameters():
        print(name + " .... ")
        print("value:")
        print(param.data.numpy())
        print("grad: ")
        print(param.grad.data.numpy())
    optimizer.step()

#print(loss.grad)
