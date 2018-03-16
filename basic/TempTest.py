import numpy as np
import torch
from torchvision.datasets import mnist
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from mnist.model.AlexNet import *
from mnist.MnistRun import *
use_cuda = True

# def data_tf(x):
#     x = np.array(x, dtype='float32')/255
#     x = (x - 0.5) / 0.5
#     x = x.reshape((1,28,28))
#     x = torch.from_numpy(x)
#     return x
#
# train_set = mnist.MNIST('../data/mnist', train=False, transform=data_tf, download=False)
# test_set = mnist.MNIST('../data/mnist', train=False, transform=data_tf, download=False)
# train_data = DataLoader(train_set, batch_size=64, shuffle=True)
# test_data = DataLoader(test_set, batch_size=128, shuffle=False)

train_data, test_data = getDataLoader()
#net = MnistAlexNet()
net = AlexNet()

def train(net):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), 0.01)

    if use_cuda:
        criterion = criterion.cuda()
        net = net.cuda()
    print("Start training now ...")
    losses = []
    acces = []
    eval_losses = []
    eval_acces = []

    for e in range(50):
        train_loss = 0
        train_acc = 0
        net.train()
        for im, label in train_data:
            # print(im.shape)
            im = Variable(im)
            label = Variable(label)
            if use_cuda:
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
        for im, label in test_data:
            im = Variable(im)
            label = Variable(label)
            if use_cuda:
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

train(net)

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(net.parameters(), 0.01)
#
# if use_cuda:
#     criterion = criterion.cuda()
#     net = net.cuda()
# print("Start training now ...")
# losses = []
# acces = []
# eval_losses = []
# eval_acces = []
#
# for e in range(50):
#     train_loss = 0
#     train_acc = 0
#     net.train()
#     for im, label in train_data:
#         #print(im.shape)
#         im = Variable(im)
#         label = Variable(label)
#         if use_cuda :
#             im = im.cuda()
#             label = label.cuda()
#         # forward
#         out = net(im)
#         loss = criterion(out, label)
#         # backward
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         train_loss += loss.data[0]
#         _, pred = out.max(1)
#         num_correct = (pred == label).sum().data[0]
#         acc = float(num_correct / float(im.shape[0]))
#         train_acc += acc
#
#     losses.append(train_loss / len(train_data))
#     acces.append(train_acc / len(train_data))
#     # evaluate in th test set
#     eval_loss = 0
#     eval_acc =0
#     net.eval()
#     for im, label in test_data:
#         im = Variable(im)
#         label = Variable(label)
#         if use_cuda :
#             im = im.cuda()
#             label = label.cuda()
#         out = net(im)
#         loss = criterion(out, label)
#         eval_loss += loss.data[0]
#         _, pred = out.max(1)
#         num_correct = (pred == label).sum().data[0]
#         acc = num_correct / float(im.shape[0])
#         eval_acc += acc
#
#     eval_losses.append(eval_loss / len(test_data) )
#     eval_acces.append(eval_acc / len(test_data) )
#
#     print("train acc: {}, train data len: {}".format(train_acc, len(train_data)))
#     print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:6f}'
#           .format(e, train_loss/len(train_data), train_acc / len(train_data),
#                   eval_loss / len(test_data), eval_acc / len(test_data)))