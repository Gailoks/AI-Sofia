import torch
import Tokens as tk
import ServiceTokens as st
from Seq2Seqmodule import *


def evaluate(encoder: Encoder, decoder: Decoder, tokenizer: tk.Tokenizer, text: str, device: str = 'cpu', out_len: int = 200):

    service = st.ServiceTokens(tokenizer.count_tokens())

    tokens = torch.LongTensor(list(tokenizer.tokenize(text))).to(device)

    encoded_hidden = encoder(tokens)

    out = decoder(encoded_hidden, out_len, device)

    out = out.argmax(1, keepdim=True).view(out_len)

    predicted_text = ""

    for x in filter(lambda x: x != service.get(st.STIO_NULL), out.tolist()):
        predicted_text += tokenizer.decode_token(x)

    return predicted_text

#question = input("Введите запрос ")
tokens = tk.TokenDictionary.load(".aistate/tokens.json")
tokenizer = tk.Tokenizer(tokens)
device = "cpu"

encoder = torch.load(".aistate/encoder.pkl").to(device)
decoder = torch.load(".aistate/decoder.pkl").to(device)
encoder.eval()
decoder.eval()

answer = evaluate(encoder, decoder, tokenizer, input(), device)
print(answer)