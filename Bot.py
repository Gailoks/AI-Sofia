from aiogram import Bot, types, Dispatcher, executor
import json
import torch
import torch.nn as nn
from collections import Counter
import Tokenizer as tk

with open("config.json") as fcc_file:
    fcc_data = json.load(fcc_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 1201

bot = Bot(fcc_data["token"])
dp = Dispatcher(bot)
tokens = tk.TokenDictionary.load('tokens.json')
if tokens.count() != VOCAB_SIZE-1:
    raise Exception("Invalid tokens dictionary!")
tokenizer = tk.Tokenizer(tokens)


class RnnTextGen(nn.Module):

    def __init__(self, input_size, inp_lstm_size, hid_size, n_layers, out_size, dropout=0.2) -> None:
        super(RnnTextGen, self).__init__()
        self.input_size = input_size
        self.n_layers = n_layers
        self.hidden_size = hid_size
        self.Encoder = nn.Embedding(input_size, inp_lstm_size)
        self.lstm = nn.LSTM(inp_lstm_size, hid_size, n_layers)
        self.dropout = nn.Dropout(dropout)
        self.l1 = nn.Linear(hid_size, out_size)

    def forward(self, x, hidden=None):
        x = self.Encoder(x)
        x, hidden = self.lstm(x)
        x = self.dropout(x)
        x = self.l1(x)
        return x, hidden

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device))


model = RnnTextGen(VOCAB_SIZE, 1000, 500, 2, VOCAB_SIZE).to(device)
model = torch.load("data.pkl").to(device)


def evaluate(model: RnnTextGen, text: str, prediction_lim: int = 15):
    text_idx = torch.LongTensor(list(tokenizer.tokenize(text))).to(device)
    hidden = model.init_hidden()
    inp = text_idx
    predicted_text = ""
    for i in range(prediction_lim):
        next_w, hidden = model(inp.view(-1, 1).to(device), hidden)
        inp = torch.cat([inp, next_w[-1].argmax().view(-1)])
        if next_w[-1].argmax() == torch.LongTensor([VOCAB_SIZE-1]).to(device):
            break
        word = tokens.decode(int(next_w[-1].argmax()))
        predicted_text += word
    return predicted_text


@dp.message_handler(lambda message: message.text)
async def anyanswer(message: types.Message):
    await message.answer(evaluate(model, message.text, 40))

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
