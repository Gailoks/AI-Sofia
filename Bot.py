from aiogram import Bot, types, Dispatcher, executor
import json
import torch
import Tokens as tk
from evaluatemodel import evaluate
from RnnTextGen import RnnTextGen

with open("config.json") as fcc_file:
    fcc_data = json.load(fcc_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bot = Bot(fcc_data["token"])
dp = Dispatcher(bot)
model = RnnTextGen.load('data.pkl').to(device)
tokens = tk.TokenDictionary.load("config.json")
tokenizer = tk.Tokenizer(tokens)


@dp.message_handler(lambda message: message.text)
async def anyanswer(message: types.Message):
    text = evaluate(model, tokenizer, torch.LongTensor(
        [len(tokens.tokens)]), tokens, message.text, 50, device)
    await message.answer(text)

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
