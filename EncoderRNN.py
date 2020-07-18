import torch.nn as nn
import torch
device = torch.device("cuda" if torch.cuda.is_available() else"cpu")

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        #self.embedding = nn.Embedding(input_size, hidden_size)
        #self.fc1 = nn.Linear(in_features= , out_features=hidden_size)
        self.gru = nn.GRU(100, hidden_size)

    def forward(self, input):# hidden):

        embedded = input.view(1, 1, -1)
        output, hidden = self.gru(embedded)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)