import torch
import torch.nn as nn
import ServiceTokens as st
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocabulary_size, input_size, hid_size, n_layers, dropout=0.2):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.input_size = input_size
        self.hidden_size = hid_size
        self.n_layers = n_layers
        self.kernel_size = 4

        self.embedding = nn.Embedding(vocabulary_size, input_size)
        self.lstm = nn.LSTM(input_size, hid_size, n_layers, dropout=dropout)
        self.conv = nn.Sequential(nn.Conv1d(self.input_size,self.input_size,self.kernel_size),
                                  nn.MaxPool1d(self.kernel_size))


    def forward(self, x, hidden=None):
        x = self.embedding(x)
        encoder_out, hidden = self.lstm(x,hidden)

        return encoder_out, hidden
    
class Decoder(nn.Module):
    def __init__(self, vocabulary_size, input_size, hid_size, **_):
        super().__init__()
        
        self.hidden_size = hid_size
        self.out_size = vocabulary_size + st.SERVICE_OUTPUT_SIZE
        self.embedding = nn.Embedding(self.out_size, input_size)

        self.lstm = nn.LSTMCell(hid_size, hid_size)
        self.output_linear = nn.Linear(hid_size, self.out_size, bias=False)

        self.relu = nn.ReLU()


    def forward(self,encoder_out,hidden=None):
        out, hidden = self.lstm(encoder_out, hidden)
        out = self.output_linear(out)
    
        #out = self.relu(out)
        out = F.log_softmax(out, dim=-1)

        return out, hidden