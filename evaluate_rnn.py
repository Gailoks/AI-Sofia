import torch
import Tokens as tk
import ServiceTokens as st
import RnnTextGen as ai

def evaluate(model:torch.nn.Module, tokenizer, text: str=None, max_length: int = 15, device:str='cpu',status:list=None):
    service = st.ServiceTokens(tokenizer.count_tokens())

    endt = torch.LongTensor([service.get(st.STO_END)])
    rolt = torch.LongTensor([service.get(st.STI_ROLE)])

    predicted_text = ""

    if status:
        inp, hidden = status
    
    else:
        text_idx = torch.LongTensor(list(tokenizer.tokenize(text))).to(device)
        hidden = model.init_hidden()
        
        

    for tensor in text_idx:
        out, hidden = model(tensor.view(-1,1), hidden)
        if out.argmax() == rolt:
            inp = rolt.to(device)
            for i in range(max_length):

                out, hidden = model(inp.view(-1,1), hidden)
                inp = out.argmax()
                if inp == rolt:
                    predicted_text+=" "
                    break
                word = tokenizer.decode_token(int(inp))
                predicted_text += word
                max_length -= len(word)
    return predicted_text

#question = input("Введите запрос ")
tokens = tk.TokenDictionary.load(".aistate/tokens.json")
tokenizer = tk.Tokenizer(tokens)
device = "cpu"

Model = torch.load("moduledata.pkl").to(device)

Model.eval()

answer = evaluate(Model, tokenizer,"что такое большие данные?что такое Распределенный реестр?",300)
print(answer)
#answer, status = evaluate(Model, tokenizer,max_length=50,status=status)
#print(answer)