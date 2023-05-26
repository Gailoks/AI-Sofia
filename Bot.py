from aiogram import Bot, types, Dispatcher, executor
import json
import torch
import torch.nn as nn
from collections import Counter
from Tokenizer import Tokenizer

with open("config.json") as fcc_file:
    fcc_data = json.load(fcc_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 1200
DEFAULT_END = "_end"
DEFAULT_PASS = " "
DEFAULT_UNK = ''
bot = Bot(fcc_data["token"])
dp = Dispatcher(bot)


a = open(r"text.txt", encoding="utf-8").read().lower()
tokenizer = Tokenizer(VOCAB_SIZE)
tokenizer.fit([a], DEFAULT_END, DEFAULT_UNK)


class RnnTextGen(nn.Module):

    def __init__(self, voc_size, inp_size, hid_size, n_layers, dropout=0.2) -> None:
        super(RnnTextGen, self).__init__()
        self.voc_size = voc_size
        self.n_layers = n_layers
        self.hidden_size = hid_size
        self.Encoder = nn.Embedding(voc_size, inp_size)
        self.lstm = nn.LSTM(inp_size, hid_size, n_layers)
        self.dropout = nn.Dropout(dropout)
        self.l1 = nn.Linear(hid_size, voc_size)

    def forward(self, x, hidden=None):
        x = self.Encoder(x)
        x, hidden = self.lstm(x)
        x = self.dropout(x)
        x = self.l1(x)
        return x, hidden

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device))


model = RnnTextGen(VOCAB_SIZE, 1000, 500, 2).to(device)
model = torch.load(r"D:\Projects\RnnTextGen\data.pkl").to(device)


def evaluate(model: torch.nn.Module, text: str, prediction_lim: int = 15):
    text_idx = torch.LongTensor(list(tokenizer.tokenize(text))).to(device)
    hidden = model.init_hidden()
    inp = text_idx
    predicted_text = ""
    for i in range(prediction_lim):
        next_w, hidden = model(inp.view(-1, 1).to(device), hidden)
        inp = torch.cat([inp, next_w[-1].argmax().view(-1)])
        word = tokenizer.rw_tokens[int(next_w[-1].argmax())]
        if next_w[-1].argmax() == torch.LongTensor([0]).to(device):
            break
        predicted_text += word
    return predicted_text


@dp.message_handler()
async def anyanswer(message: types.Message):
    await message.answer(evaluate(model, message.text, 40))

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
