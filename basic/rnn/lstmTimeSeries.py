import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable

def show_data_pic():
    data_csv = pd.read_csv('./data.csv', usecols=[1])
    plt.plot(data_csv)
    plt.show()

def normalize_data():
    data_csv = pd.read_csv('./data.csv', usecols=[1])
    data_csv = data_csv.dropna()
    dataset = data_csv.values
    dataset = dataset.astype('float32')
    max_value = np.max(dataset)
    min_value = np.min(dataset)
    scalar = max_value - min_value
    dataset = list(map(lambda x: x / scalar,dataset))
    return dataset

def create_dataset(dataset, look_back=2):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super(lstm_reg, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
        self.reg = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.rnn(x) # (seq, batch, hidden)
        s,b,h = x.shape
        x = x.view(s*b, h)
        x = self.reg(x)
        x = x.view(s,b,-1)
        return x

if __name__ == "__main__":
    #show_data_pic()
    dataset = normalize_data()
    data_X, data_Y = create_dataset(dataset)
    train_size = int(len(data_X) * 0.7)
    test_size = len(data_X) - train_size
    train_X =data_X[:train_size]
    train_Y =data_Y[:train_size]
    test_X = data_X[train_size:]
    test_Y = data_Y[train_size:]

    train_X = train_X.reshape(-1,1,2)
    train_Y = train_Y.reshape(-1,1,1)
    test_X = test_X.reshape(-1,1,2)

    train_x = torch.from_numpy(train_X)
    train_y = torch.from_numpy(train_Y)
    test_x = torch.from_numpy(test_X)

    net = lstm_reg(2,4)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=1e-2)

    for e in range(2000):
        var_x = Variable(train_x)
        var_y = Variable(train_y)
        # forward
        out = net(var_x)
        loss = criterion(out, var_y)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (e + 1) % 100 == 0:
            print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.data[0]))

    net = net.eval()
    data_X = data_X.reshape(-1,1,2)
    data_X = torch.from_numpy(data_X)
    var_data = Variable(data_X)
    pred_test = net(var_data)
    pred_test = pred_test.view(-1).data.numpy()

    plt.plot(pred_test, 'r', label='prediction')
    plt.plot(dataset, 'b', label='real')
    plt.legend(loc='best')
    plt.show()



