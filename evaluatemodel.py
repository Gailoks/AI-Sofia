import torch


def evaluate(model, tokenizer, Endt, text: str, prediction_lim: int = 15, device='cpu'):
    text_idx = torch.LongTensor(list(tokenizer.tokenize(text))).to(device)
    hidden = model.init_hidden()
    inp = text_idx
    predicted_text = ""
    Endt = Endt.to(device)
    for i in range(prediction_lim):
        next_w, hidden = model(inp.view(-1, 1).to(device), hidden)
        inp = torch.cat([inp, next_w[-1].argmax().view(-1)])
        if next_w[-1].argmax() == Endt:
            break
        word = tokenizer.decode_token(int(next_w[-1].argmax()))
        predicted_text += word
    return predicted_text
