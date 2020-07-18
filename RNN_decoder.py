import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class RNN_Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(RNN_Decoder, self).__init__()
        #self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        # self.c1 = nn.Conv1d(input_size, hidden_size, 2)
        # self.p1 = nn.AvgPool1d(2)
        # self.c2 = nn.Conv1d(hidden_size, hidden_size, 1)
        # self.p2 = nn.AvgPool1d(2)
        self.gru = nn.GRU(100, hidden_size, n_layers,batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, inputs,hidden):
        # batch_size = inputs.size(1)
        #
        # # Turn (seq_len x batch_size x input_size) into (batch_size x input_size x seq_len) for CNN
        # inputs = inputs.transpose(0, 1).transpose(1, 2)
        #
        # # Run through Conv1d and Pool1d layers
        # c = self.c1(inputs)
        # p = self.p1(c)
        # c = self.c2(p)
        # p = self.p2(c)
        #
        # # Turn (batch_size x hidden_size x seq_len) back into (seq_len x batch_size x hidden_size) for RNN
        # p = p.transpose(1, 2).transpose(0, 1)
        #
        # p = F.tanh(p)
        output, hidden = self.gru(inputs,hidden)
        #conv_seq_len = output.size(0)
        #output = output.view(conv_seq_len * batch_size,self.hidden_size)  # Treating (conv_seq_len x batch_size) as batch_size for linear layer
        #output = F.tanh(self.out(output))
        output = self.out(output)
        #output = output.view(conv_seq_len, -1, self.output_size)
        return output, hidden