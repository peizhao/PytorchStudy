from model.AlexNet import *
from torch import nn
from torch.autograd import Variable

data = Variable(torch.randn(1, 1, 28, 28))
alexNetRawNet = AlexNetRaw()
print(alexNetRawNet.features(data).size())

alexNet = AlexNet()
print(alexNet.features(data).size())