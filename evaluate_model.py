import torch
import Tokens as tk
import ServiceTokens as st
import RnnTextGen as ai

def evaluate(Encoder, Decoder, tokenizer,text: str=None, max_length: int = 15, device:str='cpu',status:list=None):
    service = st.ServiceTokens(tokenizer.count_tokens())

    endt = torch.LongTensor([service.get(st.STO_END)])
    rolt = service.get(st.STI_ROLE)

    predicted_text = ""

    if status:
        inp, hidden, encoderoutputs = status
    
    else:
        text_idx = torch.LongTensor(list(tokenizer.tokenize(text))+[rolt]).to(device)
        encoderoutputs, hidden = Encoder(text_idx)
        inp = torch.LongTensor([rolt]).to(device)
    while max_length>0:
        out, hidden = Decoder(inp.view(-1), hidden, encoderoutputs)
        inp = out.argmax()
        if inp == endt:
            break
        word = tokenizer.decode_token(int(inp))
        predicted_text += word
        max_length -= len(word)
    return predicted_text,(inp,hidden,encoderoutputs)

#question = input("Введите запрос ")
tokens = tk.TokenDictionary.load(".aistate/tokens.json")
tokenizer = tk.Tokenizer(tokens)
device = "cpu"

Encoder = torch.load("Encoderdata.pkl").to(device)
Decoder = torch.load("Decoderdata.pkl").to(device)
Encoder.eval()
Decoder.eval()

answer, status = evaluate(Encoder, Decoder, tokenizer,"что делаешь?как дела?",30)
print(answer)
answer, status = evaluate(Encoder, Decoder, tokenizer,max_length=5000,status=status)
print(answer)