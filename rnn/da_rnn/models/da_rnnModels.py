import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import logging as logger

logger.basicConfig(level=logger.INFO)

class DaRNN_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, T):
        # input size: number of underlying factors (81)
        # T: number of time steps (10)
        # hidden_size: dimension of the hidden state
        super(DaRNN_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T
        self.lstm_layer = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = 1)
        self.attn_linear = nn.Linear(in_features = 2 * hidden_size + T - 1, out_features = 1)

    def forward(self, input_data):
        # input_data: batch_size * T - 1 * input_size
        input_weighted = Variable(input_data.data.new(input_data.size(0), self.T - 1, self.input_size).zero_())
        input_encoded = Variable(input_data.data.new(input_data.size(0), self.T - 1, self.hidden_size).zero_())
        # hidden, cell: initial states with dimention hidden_size
        hidden = self.init_hidden(input_data)                     # 1 * batch_size * hidden_size
        cell = self.init_hidden(input_data)                       # 1 * batch_size * hidden_size
        # hidden.requires_grad = False
        # cell.requires_grad = False
        for t in range(self.T - 1):
            # Eqn. 8: concatenate the hidden states with each predictor
            hidden_permute = hidden.repeat(self.input_size,1,1)     # input_size * batch_size * hidden_size
            hidden_permute = hidden_permute.permute(1,0,2)          # batch_size * input_size * hidden_size
            cell_permute = cell.repeat(self.input_size,1,1)         # input_size * batch_size * hidden_size
            cell_permute = cell_permute.permute(1,0,2)              # batch_size * input_size * hidden_size
            input_data_permute = input_data.permute(0,2,1)          # batch_size * input_size * (T-1)
            x = torch.cat((hidden_permute,cell_permute,input_data_permute), dim = 2)   # batch_size * input_size * (2*hidden_size + T - 1)
            # Eqn. 9: Get attention weights
            x = x.view(-1, self.hidden_size*2 + self.T -1)          # ( (batch_size * input_size) , (2*hidden_size + T-1))
            x = self.attn_linear(x)                                 # ((batch_size * input_size) , 1)
            x = x.view(-1, self.input_size)                         #( batch_size , input_size)
            attn_weights = F.softmax(x)                             # batch_size * input_size, attn weights with values sum up to 1.
            # Eqn. 10: LSTM  ( input_data[:,t,: ] = (batch_size, input_size))
            weighted_input = torch.mul(attn_weights, input_data[:, t, :])   # batch_size * input_size
            # Fix the warning about non-contiguous memory
            # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            hidden = lstm_states[0]   # 1 * batch_size * hidden_size
            cell = lstm_states[1]   # 1 * batch_size * hidden_size
            # Save output
            input_weighted[:, t, :] = weighted_input  # batch_size * 1 * input_size
            input_encoded[:, t, :] = hidden           # batch_size * 1 * hidden_size
        return input_weighted, input_encoded       # batch_size * T-1 * input_size,  batch_size * T-1 * hidden_size

    def init_hidden(self, x):
        # No matter whether CUDA is used, the returned variable will have the same type as x.
        return Variable(x.data.new(1, x.size(0), self.hidden_size).zero_()) # dimension 0 is the batch dimension

class DaRNN_decoder(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, T):
        super(DaRNN_decoder, self).__init__()

        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.attn_layer = nn.Sequential(
            nn.Linear(2 * decoder_hidden_size + encoder_hidden_size, encoder_hidden_size),
            nn.Tanh(), nn.Linear(encoder_hidden_size, 1))
        self.lstm_layer = nn.LSTM(input_size=1, hidden_size=decoder_hidden_size)
        self.fc = nn.Linear(encoder_hidden_size + 1, 1)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, 1)
        self.fc.weight.data.normal_()

    def forward(self, input_encoded, y_history):
        # input_encoded: batch_size * T - 1 * encoder_hidden_size
        # y_history: batch_size * (T-1)
        # Initialize hidden and cell, 1 * batch_size * decoder_hidden_size
        hidden = self.init_hidden(input_encoded)           # 1 * batch_size * decoder_hidden_size
        cell = self.init_hidden(input_encoded)             # 1 * batch_size * decoder_hidden_size
        # hidden.requires_grad = False
        # cell.requires_grad = False
        for t in range(self.T - 1):
            # Eqn. 12-13: compute attention weights
            ## batch_size * T * (2*decoder_hidden_size + encoder_hidden_size)
            hidden_permute = hidden.repeat(self.T-1, 1, 1)      # T-1 * batch_size * decoder_hidden_size
            hidden_permute = hidden_permute.permute(1,0,2)      # batch_size * T-1 * decoder_hidden_size
            cell_permute = cell.repeat(self.T-1, 1, 1)          # T-1 * batch_size * decoder_hidden_size
            cell_permute = cell_permute.permute(1,0,2)          # batch_size * T-1 * decoder_hidden_size
            x = torch.cat((hidden_permute,cell_permute, input_encoded), dim = 2)      # batch_size * T-1 * (2*decoder_hidden_size + encoder_hidden_size)
            x = x.view(-1 , 2 * self.decoder_hidden_size + self.encoder_hidden_size)  # ((batch_size * T-1), (2*decoder_hidden_size + encoder_hidden_size))
            x = self.attn_layer(x)                                                    # ((batch_size * T-1), 1)
            x = x.view(-1, self.T-1)                                                  # batch_size * T-1
            x = F.softmax(x)                                                          # batch_size * T-1, row sum up to 1
            # Eqn. 14: compute context vector
            x = x.unsqueeze(1)                                                        # batch_size * 1 * T-1
            # x = (batch_size,1,T-1), input_encoded=(batch_size,T-1,encoder_hidden_size)
            contexts = torch.bmm(x, input_encoded)                                    # batch_size * 1 * encoder_hidden_size
            context = torch.bmm(x, input_encoded)[:, 0, :]                            # batch_size * encoder_hidden_size
            if t < self.T - 1:
                # Eqn. 15
                y_history_t = y_history[:,t]                                          # batch_size
                y_history_t = y_history_t.unsqueeze(1)                                # batch_size * 1
                y = torch.cat((context, y_history_t), dim = 1)                        # batch_size * (encoder_hidden_size + 1)
                y_tilde = self.fc(y)                                                  # batch_size * 1
                y_tilde = y_tilde.unsqueeze(0)                                        # 1 * batch_size * 1  LSTM input is: (seq_len, batch, input_size)
                # Eqn. 16: LSTM
                self.lstm_layer.flatten_parameters()
                result, lstm_output = self.lstm_layer(y_tilde, (hidden, cell))
                hidden = lstm_output[0]                                               # 1 * batch_size * decoder_hidden_size
                cell = lstm_output[1]                                                 # 1 * batch_size * decoder_hidden_size
        # Eqn. 22: final output
        y_pred = torch.cat((hidden[0], context), dim = 1)                             # batch_size * (decoder_hidden_size + encoder_hidden_size)
        y_pred = self.fc_final(y_pred)                                                # batch_size * 1
        return y_pred

    def init_hidden(self, x):
            return Variable(x.data.new(1, x.size(0), self.decoder_hidden_size).zero_())

if __name__ == "__main__":
    encoder_input = Variable(torch.randn(128,9,81))
    en = DaRNN_encoder(81,64,10)    # input, hidden , T
    encoder_output = en(encoder_input)
    print(encoder_output[0].shape)
    print(encoder_output[1].shape)

    history = Variable(torch.rand(128, 9))
    de = DaRNN_decoder(64, 64, 10)  # encode, hidden, T
    decode_output = de(encoder_output[1], history)
    print(decode_output.shape)
