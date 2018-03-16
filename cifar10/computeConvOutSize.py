import torch
from models.VGG import *
from torch import nn
from torch.autograd import Variable

def netFeaturesSize(net, inputShape):
    data = Variable(torch.randn(inputShape))
    print("net {} size is: ".format(net.model_name))
    print(net.features(data).size())

if __name__ == "__main__":
    shape = (1,3,32,32)
    net = VGGNet()
    netFeaturesSize(net,shape)
