import torch
import Tokens as tk
import ServiceTokens as st


def evaluate(Encoder, Decoder, tokenizer,text: str=None, max_length: int = 15, device:str='cpu',status:list=None):
    service = st.ServiceTokens(tokenizer.count_tokens())

    NULL_TOKEN = torch.LongTensor([service.get(st.STIO_NULL)]).to(device)

    predicted_text = ""

    if status:
        inp, hidden = status
    
    else:
        text_idx = torch.LongTensor(list(tokenizer.tokenize(text))).to(device)
        hidden = Encoder(text_idx)
        inp = NULL_TOKEN
    for i in range(max_length):
        out, hidden = Decoder(inp.view(-1), hidden)
        inp = out.argmax()
        if inp == NULL_TOKEN:
            continue
        word = tokenizer.decode_token(int(inp))
        predicted_text += word
    return predicted_text,(inp,hidden)

#question = input("Введите запрос ")
tokens = tk.TokenDictionary.load(".aistate/tokens.json")
tokenizer = tk.Tokenizer(tokens)
device = "cpu"

Encoder = torch.load(".aistate/encoder.pkl").to(device)
Decoder = torch.load(".aistate/decoder.pkl").to(device)
Encoder.eval()
Decoder.eval()

answer, status = evaluate(Encoder, Decoder, tokenizer,"как дела?",30)
print(answer)
answer, status = evaluate(Encoder, Decoder, tokenizer,max_length=50,status=status)
print(answer)