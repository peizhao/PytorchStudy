import torch
from torch.autograd import Variable
from torch import nn
import numpy as np

input_size = 10
output_size =20
seq_size = 5
batch_size = 2
layer_size =1

rnnNet = nn.RNN(input_size, output_size, layer_size)
lstmNet = nn.LSTM(input_size, output_size, layer_size)
gruNet = nn.GRU(input_size, output_size, layer_size)

print("The Output shape for RNN, LSTM, GRU >>>>>>")
input = Variable(torch.randn(seq_size,batch_size, input_size))
out_rnn, hidden_rnn = rnnNet(input)
out_lstm, hidden_lstm = lstmNet(input)
out_gru, hidden_gru = gruNet(input)
print("Input Seq {}, Batch {}, NUM {}".format(seq_size,batch_size,input_size))
print("RNN output shape: {}".format(out_rnn.shape))
print("LSTM output shape: {}".format(out_lstm.shape))
print("GRU output shape: {}".format(out_gru.shape))
print('\n')

print("The Hiden shape for RNN, LSTM, GRU >>>>>>")
print("Input Seq {}, Batch {}, NUM {}".format(seq_size,batch_size,input_size))
print("RNN hidden shape: {}".format(hidden_rnn.shape))
print("LSTM hidden shape: {}, cell shape: {}".format(hidden_lstm[0].shape, hidden_lstm[1].shape))
print("GRU hidden shape: {}".format(hidden_gru.shape))
print('\n')




