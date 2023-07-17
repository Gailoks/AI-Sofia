import torch.nn as nn
import torch
import numpy as np
import json
import Dataset as ds
import Tokens as tk
import ServiceTokens as st
import DatasetIterator as dsi


def train(epoches: int, model:nn.Module, device: str, tokenizer, dataset) -> None:
    """epoches - number of epoches through all dataset
    model - model required to teach
    batch_size - n/a"""
    service_tokens = st.ServiceTokens(tokenizer.count_tokens())
    dataset_iterator = dsi.DatasetIterator(dataset, service_tokens, tokenizer, device)
 

    loss_func = nn.NLLLoss()

    optim =torch.optim.Adam(model.parameters(), lr=1e-2, amsgrad=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5, verbose=True, factor=0.5)

    loss_avg = []

    model.train()

    for epoch in range(epoches):
        for sample in dataset_iterator.iterate(4,35):
            nninput = sample.question
            nnexcept = sample.exceptAnswer
            mask = sample.mask

            batch_size = nninput.shape[0]
            hidden = model.init_hidden(batch_size, device)

            outputs = torch.LongTensor([]).to(device)
            for train in nninput.permute(1,0):
                output, hidden = model(train.view(1,batch_size), hidden)
                outputs = torch.cat((outputs, output),0)


            loss = loss_func(outputs.permute(1,2,0), nnexcept)
            loss.backward()

            optim.step()

            optim.zero_grad()

            loss_avg.append(loss.item())
            if len(loss_avg) >= 15:
                mean_loss = np.mean(loss_avg)
                print(f'Loss: {mean_loss}')

                scheduler.step(mean_loss)

                loss_avg = []

    torch.save(model,'moduledata.pkl')

if __name__ == "__main__":
    from RnnTextGen import RnnTextGen

    with open("ai.json") as config:
        parametrs = json.load(config)


    tokens = tk.TokenDictionary.load(".aistate/tokens.json")
    tokenizer = tk.Tokenizer(tokens)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RnnTextGen(**parametrs).to(device)
    dataset = ds.load()

    train(15, model, device, tokenizer, dataset)
