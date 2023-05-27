import torch.nn as nn
import torch
import numpy as np
import json


def train(epoches: int, model: nn.Module, device: str, tokenizer, tokens, optimizer, scheduler, loss_func, dataset) -> None:
    """epoches - number of epoches through all dataset
    model - model required to teach
    batch_size - n/a"""
    def get_batch(dataset: list):
        for qa in dataset:
            question_idx = list(tokenizer.tokenize(qa.question))
            target = list(tokenizer.tokenize(qa.answer))+[tokens.count()]
            test = question_idx+target[:-1]

            target = torch.LongTensor(target).to(device)
            test = torch.LongTensor(test).to(device)
            yield target, test

    loss_avg = []
    model.train()
    for epoch in range(epoches):
        for target, train in get_batch(dataset):

            hidden = model.init_hidden()

            output, hidden = model(train, hidden)
            target_len = len(target)
            loss = loss_func(output[-target_len:], target)

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
    import Tokens as tk
    import DatasetLoader
    with open("ai.json") as config:
        parametrs = json.load(config)

    tokens = tk.TokenDictionary.load("tokens.json")
    tokenizer = tk.Tokenizer(tokens)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RnnTextGen.RnnTextGen(**parametrs, device=device).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        patience=5,
        verbose=True,
        factor=0.5
    )
    loss_func = nn.CrossEntropyLoss()
    train(15, model, device, tokenizer, tokens, optim,
          scheduler, loss_func, DatasetLoader.load())
    model.save('data.pkl')
