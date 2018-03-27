import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import logging as logger
from da_rnnModels import *
from prepare_data import *

logger.basicConfig(level=logger.INFO)

use_cuda = torch.cuda.is_available()
loss_func =  nn.MSELoss()
n_epoch = 300
batch_size = 128
T = 10
learning_rate = 0.001

# get the data
X,Y = getCSVDataValues(get_dataPath(data_prefix,data_file),'NDX')
train_size = int(getTrainSize(X,0.7))
Y.resize(Y.shape[0])
Y = Y - np.mean(Y[:int(train_size)])
# get the encoder and decoder

if use_cuda:
    encoder = DaRNN_encoder(81,64,T).cuda()
    decoder = DaRNN_decoder(64,64,T).cuda()
else:
    encoder = DaRNN_encoder(81, 64, T)
    decoder = DaRNN_decoder(64, 64, T)

encoder_optimizer = optim.Adam(params=itertools.ifilter(lambda p: p.requires_grad, encoder.parameters()),
                                    lr=learning_rate)
decoder_optimizer = optim.Adam(params=itertools.ifilter(lambda p: p.requires_grad, decoder.parameters()),
                                    lr=learning_rate)

def train_iteration(x, y_history, y_target):
    """
    Traning one Batch size data
    :param x:  input put x value
    :param y_history:  input target history value
    :param y_target:  input groundTrue value
    :return: loss
    """
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_weighted, input_encoded = encoder(Variable(torch.from_numpy(x).type(torch.FloatTensor).cuda()))
    y_pred = decoder(input_encoded, Variable(torch.from_numpy(y_history).type(torch.FloatTensor).cuda()))
    y_true = Variable(torch.from_numpy(y_target).type(torch.FloatTensor).cuda())              # batch_size * 1

    loss = loss_func(y_pred, y_true)
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    #logger.info("MSE: %s, loss: %s.", loss.data, (y_pred[:, 0] - y_true).pow(2).mean())
    return loss.data[0]

def predict(on_train=False):
    if on_train:
        y_pred = np.zeros(int(train_size - T + 1))
    else:
        y_pred = np.zeros(int(X.shape[0] - train_size))

    i = 0
    while i < len(y_pred):
        batch_idx = np.array(range(len(y_pred)))[i: (i + batch_size)]
        x = np.zeros((len(batch_idx), T - 1, X.shape[1]))         # batch_size * T-1 * num
        y_history = np.zeros((len(batch_idx), T - 1))             # batch_size * T-1
        # go though one batch
        for j in range(len(batch_idx)):
            if on_train:
                x[j, :, :] = X[range(batch_idx[j], batch_idx[j] + T - 1), :]
                y_history[j, :] = Y[range(batch_idx[j], batch_idx[j] + T - 1)]
            else:
                x[j, :, :] = X[range(batch_idx[j] + train_size - T, batch_idx[j] + train_size - 1),:]
                y_history[j, :] = Y[
                    range(batch_idx[j] + train_size - T, batch_idx[j] + train_size - 1)]
        y_history = Variable(torch.from_numpy(y_history).type(torch.FloatTensor).cuda())
        _, input_encoded =  encoder(Variable(torch.from_numpy(x).type(torch.FloatTensor).cuda()))
        y_pred[i:(i + batch_size)] = decoder(input_encoded, y_history).cpu().data.numpy()[:, 0]
        i += batch_size
    return y_pred

def train(epoch):
    iter_per_epoch = int(np.ceil(train_size * 1. / batch_size))
    iter_losses = np.zeros(epoch * iter_per_epoch)
    epoch_losses = np.zeros(epoch)
    logger.info("Epoch Count: %d, Iterations per epoch: %d.", epoch, iter_per_epoch)

    n_iter = 0
    for i in range(epoch):
        logger.info("epoch : {}/{}".format(i, epoch))
        perm_idx = np.random.permutation(int(train_size -T))
        j = 0
        while j < train_size:
            batch_idx = perm_idx[j:(j + batch_size)]                  # batch_size
            XValue = np.zeros((len(batch_idx), T - 1, X.shape[1]))    # batch_size * T-1 * X num
            Y_history = np.zeros((len(batch_idx), T - 1))             # batch_size * T-1
            Y_target = Y[batch_idx + T]                               # batch_size * 1

            # go through the batch index
            for k in range(len(batch_idx)):
                XValue[k, :, :] = X[batch_idx[k]: (batch_idx[k] + (T - 1)), :]   # 1 * T-1 * X num
                Y_history[k, :] = Y[batch_idx[k]: (batch_idx[k] + (T - 1))]      # 1 * T-1 * Y num
            # Now XValue = batch_size * T-1 * X num, Y_history =  batch_size * T-1 * Y num
            loss = train_iteration(XValue, Y_history, Y_target)
            iter_losses[i * iter_per_epoch + j / batch_size] = loss
            j += batch_size
            n_iter += 1

            if n_iter % 10000 == 0 and n_iter > 0:
                for param_group in encoder_optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9
                for param_group in decoder_optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9

        epoch_losses[i] = np.mean(iter_losses[range(i * iter_per_epoch, (i + 1) * iter_per_epoch)])

        if i % 10 == 0:
            logger.info("Epoch %d, loss: %3.3f.", i, epoch_losses[i])

        if i % 10 == 0:
            y_train_pred = predict(on_train=True)                       # [train_size]
            y_test_pred = predict(on_train=False)                       # [test_size]
            y_pred = np.concatenate((y_train_pred, y_test_pred))        # [total data size]
            plt.figure()
            plt.title("iter:{}".format(i))
            plt.plot(range(1, 1 + len(Y)), Y, label="True")             # Ground True
            plt.plot(range(T, len(y_train_pred) + T), y_train_pred, label='Predicted - Train')
            plt.plot(range(T + len(y_train_pred), len(Y) + 1), y_test_pred, label='Predicted - Test')
            plt.legend(loc='upper left')
    plt.show()

# plt.figure()
# plt.plot(range(1, 1 + len(Y)), Y, label="True")
# plt.show()

train(30)
