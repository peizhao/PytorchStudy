import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

model_file = '/home/leon/temp/leon/imagenet/checkpoint.pth.tar'
temp_file = './model.pt'

def loadModel():
    mm = models.__dict__['resnet18']()
    torch.save(mm, temp_file)
    nn = torch.load(temp_file)
    nn.cuda()
    os.remove(temp_file)

def loadModelDict():
    mm = models.__dict__['resnet18']()
    torch.save(mm.state_dict(), temp_file)
    nn = models.__dict__['resnet18']()
    nn.load_state_dict(torch.load(temp_file))
    nn.cuda()
    os.remove(temp_file)

def loadModelDataParallel():
    mm = models.__dict__['resnet18']()
    mm = torch.nn.DataParallel(mm).cuda()
    torch.save(mm, temp_file)
    nn = torch.load(temp_file)
    nn.cuda()
    os.remove(temp_file)

def loadModelDictDataParallel():
    mm = models.__dict__['resnet18']()
    mm = torch.nn.DataParallel(mm).cuda()
    torch.save(mm.state_dict(), temp_file)
    nn = models.__dict__['resnet18']()
    nn = torch.nn.DataParallel(nn).cuda()   # you shoulde be the same parallel model
    nn.load_state_dict(torch.load(temp_file))
    nn.cuda()
    os.remove(temp_file)

if __name__ == "__main__":
    loadModel()
    print('loadModel done')
    loadModelDict()
    print('loadModelDict done')
    loadModelDataParallel()
    print('loadModelDataParallel done')
    loadModelDictDataParallel()
    print('loadModelDictDataParallel done')
