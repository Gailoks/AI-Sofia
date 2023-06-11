import torch
import torch.nn as nn
import ServiceTokens as st


class RnnTextGen(nn.Module):

    def __init__(self, vocabulary_size, input_size, hid_size, n_layers, dropout=0.2, device ='cpu') -> None:
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hid_size
        self.out_size =vocabulary_size+st.SERVICE_OUTPUT_SIZE
        self.Encoder = nn.Embedding(vocabulary_size+st.SERVICE_INPUT_SIZE, input_size)
        self.lstm = nn.LSTM(input_size, hid_size, n_layers, dropout = dropout)
        self.l1 = nn.Linear(hid_size,self.out_size)
        self.device = device

    def forward(self, x, hidden=None):
        x = self.Encoder(x).detach()
        x, hidden = self.lstm(x, hidden)
        x = x[-1]
        x = self.l1(x)
        return x, hidden
    
    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(self.device),
               torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(self.device))

    def save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path):
        return torch.load(path)
