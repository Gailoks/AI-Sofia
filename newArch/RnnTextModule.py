import torch.nn as nn
import torch

class RnnTextModule(nn.Module):
    def __init__(self, **options):
        super().__init__()
        inp_lstm_size =  options["inp_lstm_size"]

        self.n_layers = options["n_layers"]
        self.hidden_size = options["hidden_size"]
        self.Encoder = nn.Embedding(options["input_size"], inp_lstm_size)
        self.lstm = nn.LSTM(inp_lstm_size, self.hidden_size, self.n_layers)
        self.dropout = nn.Dropout(options["dropout"])
        self.l1 = nn.Linear(self.hidden_size, options["out_size"])
        self.device = options["device"]


    def forward(self, x, hidden = None):
        x = self.Encoder(x)
        x, hidden = self.lstm(x)
        x = self.dropout(x)
        x = self.l1(x)
        return x, hidden

    def create_empty_hidden(self, batch_size):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(self.device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(self.device))