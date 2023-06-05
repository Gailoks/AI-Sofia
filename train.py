import torch.nn as nn
import torch
import numpy as np
import json
import Dataset as ds
import Tokens as tk
import ServiceTokens as st


def train(epoches: int, model: nn.Module, device: str, tokenizer, dataset) -> None:
    """epoches - number of epoches through all dataset
    model - model required to teach
    batch_size - n/a"""
    service_tokens = st.ServiceTokens(tokenizer.count_tokens())
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=5,
        verbose=True,
        factor=0.5
    )

    def get_batch(dialog: ds.Dialog):
        nninput = []
        for qa in dialog.listPairs():
            nninput += list(tokenizer.tokenize(qa.question)) + [service_tokens.get(st.STI_ROLE)]
            nnexcept = list(tokenizer.tokenize(qa.answer)) + [service_tokens.get(st.STO_END)]
            yield nninput.copy(), nnexcept.copy()
            nninput += list(tokenizer.tokenize(qa.answer)) + [service_tokens.get(st.STI_ROLE)]

    loss_avg = []
    model.train()
    for epoch in range(epoches):
        for dialog in dataset.listDialogs():
            for nninput, nnexcept in get_batch(dialog):
                output, hidden = model(torch.LongTensor(nninput).to(device))
                exceptTensor = torch.zeros(tokenizer.count_tokens() + st.SERVICE_OUTPUT_SIZE).to(device)
                exceptTensor[nnexcept[0]] = 1.0
                loss = loss_func(output, exceptTensor)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_avg.append(loss.item())

                for target in nnexcept[1::]:
                    output, hidden = model(output.argmax().view(-1), hidden)
                    exceptTensor = torch.zeros(tokenizer.count_tokens() + st.SERVICE_OUTPUT_SIZE).to(device)
                    exceptTensor[target] = 1.0

                    loss = loss_func(output, exceptTensor)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    loss_avg.append(loss.item())
                if len(loss_avg) >= 50:
                    mean_loss = np.mean(loss_avg)
                    print(f'Loss: {mean_loss}')
                    scheduler.step(mean_loss)
                    loss_avg = []


if __name__ == "__main__":
    import RnnTextGen

    with open("ai.json") as config:
        parametrs = json.load(config)

    tokens = tk.TokenDictionary.load(".aistate/tokens.json")
    tokenizer = tk.Tokenizer(tokens)
    tokenizer.count_tokens
    device = 'cpu'#torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RnnTextGen.RnnTextGen(**parametrs, device=device).to(device)
    dataset = ds.load()

    train(40, model, device, tokenizer, dataset)
    model.save('data.pkl')
