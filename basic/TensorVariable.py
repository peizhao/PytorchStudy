import torch
from torch.autograd import Variable

x_tensor = torch.randn(10,5)
y_tensor = torch.randn(10,5)
print "x_tensor is: "
print(x_tensor)
print "y_tensor is: "
print(x_tensor)
x = Variable(x_tensor, requires_grad = True)
y = Variable(y_tensor, requires_grad = True)
z = torch.sum(x+y)
#z = torch.add(x,y)
print "z data is"
print(z.data)
print "z grad_fn is "
print(z.grad_fn)
z.backward()
#z.backward(torch.ones_like(z))
print " x grad is : "
print(x.grad)
print " y grad is : "
print(y.grad)
