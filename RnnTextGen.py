import torch
import torch.nn as nn


class RnnTextGen(nn.Module):

    def __init__(self, vocabulary_size, input_size, hid_size, n_layers, out_size, device, dropout=0.2) -> None:
        super(RnnTextGen, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hid_size
        self.out_size = out_size
        self.Encoder = nn.Embedding(vocabulary_size, input_size)
        self.l1 = nn.Linear(input_size, out_size)
        self.Attention = nn.MultiheadAttention(out_size, out_size)
        self.lstm = nn.LSTM(input_size, hid_size, n_layers)
        self.dropout = nn.Dropout(dropout)
        self.l2 = nn.Linear(hid_size, out_size)
        self.device = device

    def forward(self, x, hidden=None):
        x = self.Encoder(x)
        p = self.l1(x)
        aw, _ = self.Attention(p.view(-1, self.out_size), p.view(-1, self.out_size),
                               p.view(-1, self.out_size))  # a - attn output b - attn_wheights)
        x, hidden = self.lstm(x)
        x = self.dropout(x)
        x = self.l2(x)
        x = torch.cat((aw, x))
        return x, hidden

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(self.device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(self.device))

    def save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path):
        return torch.load(path)
