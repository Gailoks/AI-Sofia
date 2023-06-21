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
    rolt = torch.LongTensor([service_tokens.get(st.STI_ROLE)]).to(device)

    loss_func = nn.NLLLoss()

    optim =torch.optim.Adam(model.parameters(), lr=1e-2, amsgrad=True)


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5, verbose=True, factor=0.5)

    loss_avg = []

    model.train()

    for epoch in range(epoches):
        for sample in dataset_iterator.iterate(1):
            nninput = sample.question
            nnexcept = sample.exceptAnswer

            encoderout, (cx,tx) = model(nninput)
            cx = cx.reshape(model.n_layers, 1, model.hidden_size)
            tx = tx.reshape(model.n_layers, 1, model.hidden_size)
            output, hidden = model(rolt.view(-1, 1), (cx, tx))
            outputs = output

            for train, target in zip(nnexcept[:-1:],nnexcept[1::]):
                output, hidden = model(train.view(-1,1), hidden)
                outputs=torch.cat((outputs, output))

            loss = loss_func(outputs.view(-1,model.out_size),nnexcept)
            loss.backward()

            optim.step()

            optim.zero_grad()

            loss_avg.append(loss.item())
            if len(loss_avg) >= 50:
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

    train(20, model, device, tokenizer, dataset)
