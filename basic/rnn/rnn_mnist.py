import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from torchvision.datasets import mnist

from torchvision import transforms as tfs
from torchvision.datasets import MNIST
from datetime import datetime

class rnn_classify(nn.Module):
    def __init__(self, in_features = 28, hidden_features=100, num_class=10, num_layers=2):
        super(rnn_classify, self).__init__()
        self.rnn = nn.LSTM(in_features, hidden_features, num_layers)
        self.classifier = nn.Linear(hidden_features, num_class)

    def forward(self, x):
        x = x.squeeze()  # (batch, 1, 28, 28) -> (batch, 28, 28)
        x = x.permute(2,0,1) # (batch, 28, 28) -> (28, batch, 28)
        out, _ = self.rnn(x)  # out: (28, batch, hidden_features)
        out = out[-1, :, :]  # out = last seq (batch, hidden_features)
        out = self.classifier(out)
        return out

class Config:
    def __init__(self, epoch, lr, use_cuda = True):
        self.epoch = epoch
        self.lr =lr
        self.use_cuda = use_cuda

def data_tf(x):
    x = np.array(x, dtype='float32')/255
    x = (x - 0.5) / 0.5
    x = x.reshape((1,28,28))
    x = torch.from_numpy(x)
    return x

def getDataLoader():
    train_set = mnist.MNIST('../../data/mnist', train=False, transform=data_tf, download=False)
    test_set = mnist.MNIST('../../data/mnist', train=False, transform=data_tf, download=False)
    train_data = DataLoader(train_set, batch_size=64, shuffle=True)
    test_data = DataLoader(test_set, batch_size=128, shuffle=False)
    return train_data, test_data

def train(net, config):
    # get the data
    train_data, test_data = getDataLoader()
    # loss function and the SGD
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(net.parameters(), config.lr)
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
    net = rnn_classify()
    config = Config(20,1e-1)
    train(net, config)