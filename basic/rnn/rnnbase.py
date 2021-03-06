import torch
from torch.autograd import Variable
from torch import nn
import numpy as np

print("Basic Ops of RNNCell ...")
rnn_single = nn.RNNCell(input_size = 100, hidden_size=200)
print("the learnable input-hidden weights shape")
print(rnn_single.weight_ih.shape)
print("the learnable input-hidden bias shape")
print(rnn_single.bias_ih.shape)
print("the learnable hidden-hidden weights shape")
print(rnn_single.weight_hh.shape)
print("the learnable hidden-hidden bias")
print(rnn_single.bias_hh.shape)

x = Variable(torch.randn(6,5,100)) # input
h_t = Variable(torch.zeros(5,200))
out = []
for i in range(6):
    h_t = rnn_single(x[i], h_t)
    print(h_t.shape)
    out.append(h_t)

print("Basic Ops of RNN ...")
rnn_seq = nn.RNN(100, 200)
print(rnn_seq.weight_ih_l0.shape)
print(rnn_seq.bias_ih_l0.shape)
print(rnn_seq.weight_hh_l0.shape)
print(rnn_seq.bias_hh_l0.shape)

out, h_t = rnn_seq(x)
print("out put shape")
print(out.shape)
print("Hiden layer state")
print(h_t.shape)

print("Basic Ops of LSTM ...")
lstm_seq = nn.LSTM(50, 100, num_layers=3)
print("layer 0: ")
print(lstm_seq.weight_ih_l0.shape)
print(lstm_seq.bias_ih_l0.shape)
print(lstm_seq.weight_hh_l0.shape)
print(lstm_seq.bias_hh_l0.shape)
print("layer 1: ")
print(lstm_seq.weight_ih_l1.shape)
print(lstm_seq.bias_ih_l1.shape)
print(lstm_seq.weight_hh_l1.shape)
print(lstm_seq.bias_hh_l1.shape)
print("layer 2:")
print(lstm_seq.weight_ih_l2.shape)
print(lstm_seq.bias_ih_l2.shape)
print(lstm_seq.weight_hh_l2.shape)
print(lstm_seq.bias_hh_l2.shape)

print("input shape is (1,1,50)")
x = Variable(torch.randn(1,1,50)) # input
lstm_seq_out, _ = lstm_seq(x)
print("lstm_seq_out shape is {}".format(lstm_seq_out.shape))

print("input shape is (3,1,50)")
x = Variable(torch.randn(3,1,50)) # input
lstm_seq_out, _ = lstm_seq(x)
print("lstm_seq_out shape is {}".format(lstm_seq_out.shape))

print("Basic Ops of Embedding ...")
embedding = nn.Embedding(10, 3)   #  the inner matrix is 12x5
input = Variable(torch.LongTensor([[1,2,4,5],[4,3,2,9]])) # the 1,2,4,5 rows and 4,3,2,9 rows
output = embedding(input)
print("result is")
print(output)
print("Embedding param: ")
for name, param in embedding.named_parameters():
    print("{} : {}".format(name, param.shape))
    print(param)

embedding = nn.Embedding(12, 5, padding_idx=0)   # the inner matrix is 12x5
for name, param in embedding.named_parameters():
    print("{} : {}".format(name, param.shape))
    print(param)
input = Variable(torch.LongTensor([[0, 2, 0, 5]]))  # the 0th,2th,0th,5th row of the inner 12x5 matrix
output = embedding(input)
print("result is:")
print(output)

print("Basic Ops of GRU ...")
grnNet = nn.GRU(10,20,4)
input =Variable(torch.randn(5,3,10)) # (seq, batch, input_size)
output,h_n = grnNet(input)
print("GRU output shape is {}".format(output.shape))
print("GRU h_n shape is {}".format(h_n.shape))
for name, param in grnNet.named_parameters():
    print("{} : {}".format(name, param.shape))

print("LSTM vs GRU ...")
gruNet = nn.GRU(10,20,2)
lstmNet = nn.LSTM(10,20,2)
input = Variable(torch.randn(1,1,10))
gruOut, gruHiden = grnNet(input)
lstmOut, lstmHiden = lstmNet(input)
print("gruOut shape: {}  gruHiden Shape: {}".format(gruOut.shape, gruHiden.shape))
print("lstmOut shape: {}  lstmHiden Shape: {}".format(lstmOut.shape, lstmHiden[0].shape))

print("Hidden Output Shape When Sequence input ...")
rnnNet = nn.RNN(10,20,1)
gruNet = nn.GRU(10,20,1)
lstmNet = nn.LSTM(10,20,1)
input = Variable(torch.randn(6,1,10))    # seq=6, batch=1, input_num =10
rnnOut, rnnHiden = rnnNet(input)
gruOut, gruHiden = grnNet(input)
lstmOut, lstmHiden = lstmNet(input)
print("rnnOut shape: {}  runnHiden Shape: {}".format(rnnOut.shape, rnnHiden.shape))
print("gruOut shape: {}  gruHiden Shape: {}".format(gruOut.shape, gruHiden.shape))
print("lstmOut shape: {}  lstmHiden Shape: {}".format(lstmOut.shape, lstmHiden[0].shape))

print("input Fix-length sequence, output Fix length sequence")
rnnNet = nn.RNN(10,20,1)
gruNet = nn.GRU(10,20,1)
lstmNet = nn.LSTM(10,20,1)
input = Variable(torch.randn(6,1,10))    # seq=6, batch=1, input_num =10
rnnOut, _ = rnnNet(input)
gruOut, _ = grnNet(input)
lstmOut, _ = lstmNet(input)
print("rnnOut shape: {} ".format(rnnOut.shape))
print("gruOut shape: {} ".format(gruOut.shape))
print("lstmOut shape: {} ".format(lstmOut.shape))

print("input vector, output sequence")
rnnNet = nn.RNN(10,20,1)
input = Variable(torch.randn(1,1,10))
output= np.zeros((6,1,20))
output = Variable(torch.from_numpy(output))
for i in range(6):
    out, _ = rnnNet(input)
    output[i,:,:] = out
print(output[1,:,:])
