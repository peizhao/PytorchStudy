import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


def vgg_block(num_convs, in_channels, out_channels):
    """
    :param num_convs: the count of the convs in this block
    :param in_channels:  the number of the input channels
    :param out_channels:  the number of the ouput chanels
    :return: net block
    """
    net = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(True)]  # the first layer
    for i in range(num_convs-1):
        net.append(nn.Conv2d(out_channels,out_channels, kernel_size=3,padding=1))
        net.append(nn.ReLU(True))
    net.append(nn.MaxPool2d(2,2))
    return nn.Sequential(*net)

def vgg_stack(num_convs, channels):
    """
    :param num_convs: the count of convs
    :param channels: array of [input_channels, output_channels]
    :return:
    """
    net = []
    for n, c in zip(num_convs, channels):
        in_channels = c[0]
        out_channels = c[1]
        net.append(vgg_block(n, in_channels, out_channels))
    return nn.Sequential(*net)

class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet,self).__init__()
        self.model_name = "VGGNet"
        self.features = vgg_stack((1, 1, 2, 2, 2), ((3, 64), (64, 128), (128, 256), (256, 512), (512, 512)))
        self.fc = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(True),
            nn.Linear(100,10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

if __name__ == "__main__":
    vgg_net = vgg_stack((1, 1, 2, 2, 2), ((3, 64), (64, 128), (128, 256), (256, 512), (512, 512)))
    print(vgg_net)

    data = torch.randn(1,3,32,32)
    data = Variable(data)
    net = VGGNet()
    x = net.features(data)
    print("x shape: {}".format(x.shape))
    x = x.view(x.shape[0], -1)
    print("x shape after view: {}".format(x.shape))

    for name, param in net.named_parameters():
        print(name)
        print(param.shape)
