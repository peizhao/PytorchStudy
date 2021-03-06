import numpy as np
import torch
from torchvision.datasets import mnist
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from common import *
from models import AlexNet

def train(net, config):
    # get the data
    train_data, test_data = getDataLoader()
    # loss function and the SGD
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), config.lr, weight_decay=1e-4)
    if(config.use_cuda):
        net = net.cuda()
        criterion = criterion.cuda()

    print("Start training now ...")
    losses = []
    acces = []
    eval_losses = []
    eval_acces = []

    for e in range(config.epoch):
        train_loss = 0
        train_acc = 0
        net.train()
        # training phase
        for im, label in train_data:
            # print(im.shape)
            im = Variable(im)
            label = Variable(label)
            if config.use_cuda:
                im = im.cuda()
                label = label.cuda()
            # forward
            out = net(im)
            loss = criterion(out, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.data[0]
            _, pred = out.max(1)
            num_correct = (pred == label).sum().data[0]
            acc = float(num_correct / float(im.shape[0]))
            train_acc += acc

        losses.append(train_loss / len(train_data))
        acces.append(train_acc / len(train_data))
        # evaluate in th test set
        eval_loss = 0
        eval_acc = 0
        net.eval()
        # test phase
        for im, label in test_data:
            im = Variable(im)
            label = Variable(label)
            if config.use_cuda:
                im = im.cuda()
                label = label.cuda()
            out = net(im)
            loss = criterion(out, label)
            eval_loss += loss.data[0]
            _, pred = out.max(1)
            num_correct = (pred == label).sum().data[0]
            acc = num_correct / float(im.shape[0])
            eval_acc += acc

        eval_losses.append(eval_loss / len(test_data))
        eval_acces.append(eval_acc / len(test_data))

        print("train acc: {}, train data len: {}".format(train_acc, len(train_data)))
        print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:6f}'
              .format(e, train_loss / len(train_data), train_acc / len(train_data),
                      eval_loss / len(test_data), eval_acc / len(test_data)))

if __name__ == "__main__":
    cf = Config(50, 0.01, True)
    net = AlexNet.AlexNet()
    train(net,cf)