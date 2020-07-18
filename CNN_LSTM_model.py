import torch.nn as nn
import torch
import random

class CNN_LSTM(nn.Module):

    def __init__(self,CNN_encoder, RNN_decoder):

        super().__init__()

        self.encoder = CNN_encoder
        self.decoder = RNN_decoder


        assert self.encoder.hidden_size == self.decoder.hidden_size, "Hidden dimensions of encoder and decoder must be equal!"
        #assert self.encoder.n_layers == self.decoder.n_layers, "Encoder and decoder must have equal number of layers!"


    def forward(self, src):

        """
        :param src : [batch_size, sent len, embedding_dim]
        :return:
               outputs : [sent len, batch size, output dim]
        """



        hidden = self.encoder(src)
        outputs,hidden = self.decoder(src,hidden)#hidden argumment is not given here
        return outputs,hidden