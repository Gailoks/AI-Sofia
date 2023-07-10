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

        self.embedding = nn.Embedding(vocabulary_size, input_size)
        self.lstm = nn.LSTM(input_size, hid_size, n_layers, dropout=dropout)


    def forward(self, x, hidden=None):
        x = self.embedding(x)
        _, hidden = self.lstm(x,hidden)

        return hidden
    
class Decoder(nn.Module):
    def __init__(self, vocabulary_size, input_size, hid_size, n_layers, dropout=0.2):
        super().__init__()
        
        self.hidden_size = hid_size
        self.out_size = vocabulary_size + st.SERVICE_OUTPUT_SIZE

        self.lstm = nn.LSTM(0, hid_size, n_layers, dropout=dropout)
        self.output_linear = nn.Linear(hid_size, self.out_size)

        self.relu = nn.ReLU()


    def forward(self, encoded_data, out_len: int, device):
        batch_size = encoded_data[0].size()[1] if len(encoded_data[0].size()) == 3 else -1

        if batch_size != -1:
            null_input_for_lstm = torch.Tensor().view(out_len, batch_size, 0).to(device)
        else:
            null_input_for_lstm = torch.Tensor().view(out_len, 0).to(device)

        out, _ = self.lstm(null_input_for_lstm, encoded_data)
        out = self.output_linear(out)
        
        out = self.relu(out)
        out = F.log_softmax(out, dim=-1)

        #if batch_size != -1:
        #    out = out.argmax(2, keepdim=True).view(out_len, batch_size)
        #else:
        #    out = out.argmax(1, keepdim=True).view(out_len)

        return out

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
