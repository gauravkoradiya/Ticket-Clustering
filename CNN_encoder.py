import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class CNN_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size,output_size):
        super(CNN_Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        #self.n_layers = n_layers

        self.c1 = nn.Conv1d(input_size, hidden_size, kernel_size=4)
        self.p1 = nn.MaxPool1d(3)
        self.c2 = nn.Conv1d(hidden_size, hidden_size,kernel_size=3)
        self.p2 = nn.MaxPool1d(2)
        #self.c3 = nn.Conv1d(hidden_size,hidden_size,kernel_size=2)
        #self.p3 = nn.MaxPool1d(2)

        #self.gru = nn.GRU(hidden_size,hidden_size, n_layers,batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, src):

        """
        :param src: [batch_size,seq_len,embedding_dim]
        :return:

        """
        #batch_size = inputs.size(1)

        # Turn (seq_len x batch_size x input_size) into (batch_size x input_size x seq_len) for CNN
        #inputs = inputs.transpose(0, 1).transpose(1, 2)
        src = src.permute(0,2,1)
        # Run through Conv1d and Pool1d layers

        c = self.c1(src)
        p = self.p1(c)
        # while p.size(2) != 1:
        #     c = self.c2(p)
        #     p = self.p2(c)
        #
        c = self.c2(p)
        p = self.p2(c)

        p=p.permute(0,2,1).squeeze()
        output = self.out(p)
        #output, hidden_state = self.gru(p)
        # Turn (batch_size x hidden_size x seq_len) back into (seq_len x batch_size x hidden_size) for RNN
        # p=F.tanh(p)
        #add a RNN unit for generate a hidden state
        # output,hidden_state = self.gru(p)

        return output