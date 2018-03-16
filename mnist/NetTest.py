from model import *
from common import *
import model.AlexNet as AlexNet
import model.FCNet as FCNet

def getFCNet_Config():
    cf = Config(20, 0.1, False)
    net = FCNet.FCNet()
    return net, cf

def getAlexNet_Config():
    cf = Config(50, 0.01, True)
    net = AlexNet.AlexNet()
    return net, cf