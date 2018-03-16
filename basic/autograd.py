import torch
from torch.autograd import Variable

x = Variable(torch.Tensor([2]), requires_grad = True)
print x
y = x + 2
z = y ** 2 + 3
print(z)

#y.backward()
z.backward()
print(x.grad)

print "array derivation ................"
m = Variable(torch.FloatTensor([[2,3]]), requires_grad = True)
n = Variable(torch.zeros(1, 2))
print(m)
print(n)
n[0,0] = m[0,0] ** 2
n[0,1] = m[0,1] ** 3
print(n)
n.backward(torch.ones_like(n))
print(m.grad)

print "one value derivation multi times ............"
x = Variable(torch.Tensor([3]), requires_grad = True)
y = x ** 2
y.backward(retain_graph = True)
print(x.grad)
y.backward()
print(x.grad)