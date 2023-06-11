import torch
import Tokens as tk
import ServiceTokens as st
import RnnTextGen as ai

def evaluate(model, tokenizer, Endt,rolt, text: str, max_length: int = 15, device='cpu'):
    text_idx = torch.LongTensor(list(tokenizer.tokenize(text))+[rolt]).to(device) 
    predicted_text = ""
    Endt = Endt.to(device)
    out,hidden = model(text_idx)
    inp = out.argmax()
    if inp == Endt:
        return "Error no message!"
    word = tokenizer.decode_token(int(inp))
    predicted_text += word
    max_length -= len(word)
    while max_length>0:
        out, hidden = model(inp.view(-1), hidden)
        inp = out.argmax()
        if inp == Endt:
            break
        word = tokenizer.decode_token(int(inp))
        predicted_text += word
        max_length -= len(word)
    return predicted_text,(inp,hidden)

question = input("Введите запрос ")
tokens = tk.TokenDictionary.load(".aistate/tokens.json")
tokenizer = tk.Tokenizer(tokens)
device = "cpu"
model = ai.RnnTextGen.load('data.pkl').to(device)
service = st.ServiceTokens(tokenizer.count_tokens())
endt = torch.LongTensor([service.get(st.STO_END)])
rolt = service.get(st.STI_ROLE)
print(evaluate(model,tokenizer,endt,rolt,question,30)[0])
