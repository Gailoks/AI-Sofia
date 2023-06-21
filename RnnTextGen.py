import torch
import torch.nn as nn
import ServiceTokens as st


class RnnTextGen(nn.Module):

    def __init__(self, vocabulary_size, input_size, hid_size, n_layers, dropout=0.2) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hid_size
        self.out_size =vocabulary_size+st.SERVICE_OUTPUT_SIZE
        self.Encoder = nn.Embedding(vocabulary_size+st.SERVICE_INPUT_SIZE, input_size)
        self.lstm = nn.LSTM(input_size, hid_size, n_layers, dropout = dropout)
        self.l1 = nn.Linear(hid_size,self.out_size)
        self.logsoftmax = nn.LogSoftmax(-1)

    def forward(self, x, hidden=None):
        x = self.Encoder(x)
        out, hidden = self.lstm(x, hidden)
        out = self.l1(out)
        out = self.logsoftmax(out)
        return out, hidden
