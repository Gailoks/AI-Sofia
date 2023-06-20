import torch
import torch.nn as nn
import ServiceTokens as st
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocabulary_size, input_size, hid_size, n_layers, dropout=0.2 ):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.input_size = input_size
        self.hidden_size = hid_size
        self.n_layers = n_layers
        self.Embedding = nn.Embedding(vocabulary_size+st.SERVICE_INPUT_SIZE, input_size)
        self.lstm = nn.LSTM(input_size,hid_size,n_layers,dropout=dropout)

    def forward(self, x, hidden=None):
        x = self.Embedding(x)
        x , hidden = self.lstm(x,hidden)
        x = sum(x)
        return x, hidden
    
class Decoder(nn.Module):
    def __init__(self, vocabulary_size, input_size, hid_size, n_layers, dropout=0.2 ):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.input_size = input_size
        self.hidden_size = hid_size
        self.out_size = vocabulary_size+st.SERVICE_OUTPUT_SIZE
        self.Embedding = nn.Embedding(vocabulary_size+st.SERVICE_INPUT_SIZE, input_size)
        self.lstm = nn.LSTM(input_size, hid_size, n_layers, dropout=dropout)
        self.l1 = nn.Linear(hid_size,self.out_size)
        self.fc_hidden = nn.Linear(hid_size,hid_size,False)
        self.fc_encoder = nn.Linear(hid_size,hid_size,False)
        self.l2 = nn.Linear(hid_size,input_size,False)

    def forward(self, x, hidden,encoder_outputs):
        embedded = self.Embedding(x)
        context_vector = self.l2(encoder_outputs)
        context_vector = F.softmax(context_vector,dim=-1)
        out = embedded+context_vector
        out, hidden = self.lstm(out, hidden)
        out = self.l1(out)
        out = F.log_softmax(out,dim=-1)
        return out, hidden
    