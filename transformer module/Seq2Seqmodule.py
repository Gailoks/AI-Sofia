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
        self.Embedding = nn.Embedding(vocabulary_size, input_size)
        self.lstm = nn.LSTM(input_size,hid_size,n_layers,dropout=dropout)


    def forward(self, x, hidden=None):
        x = self.Embedding(x)
        x , hidden = self.lstm(x,hidden)

        return hidden
    
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
        self.relu = nn.ReLU()


    def forward(self, x, hidden):
        embedded = self.Embedding(x)
        out, hidden = self.lstm(embedded, hidden)
        out = self.l1(out)
        #out = self.relu(out)
        out = F.log_softmax(out, dim=-1)
        return out, hidden

class Seq2SeqTransformer(nn.Module):
    def __init__(self, decoder:Decoder):
        self.decoder = decoder
    def forward(self, x, hidden, prediction_lim):
        out = x 
        outs = []
        for i in range(prediction_lim):
            weights, hidden = self.decoder(out, hidden)
            out = weights.argmax()
            outs += out 
        return outs, hidden
