import torch.nn as nn
import torch
import numpy as np
import json
import Dataset as ds
import Tokens as tk
import ServiceTokens as st
import DatasetIterator as dsi


def train(epoches: int, Encoder:nn.Module, Decoder:nn.Module, device: str, tokenizer, dataset) -> None:
    """epoches - number of epoches through all dataset
    model - model required to teach
    batch_size - n/a"""
    service_tokens = st.ServiceTokens(tokenizer.count_tokens())
    dataset_iterator = dsi.DatasetIterator(dataset, service_tokens, tokenizer, device)
    rolt = torch.LongTensor([service_tokens.get(st.STI_ROLE)]).to(device)

    loss_func = nn.NLLLoss()

    Encoderoptim = torch.optim.Adamax(Encoder.parameters(), lr=1e-2)
    Decoderoptim =torch.optim.Adam(Decoder.parameters(), lr=1e-2, amsgrad=True)

    escheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Encoderoptim, patience=5, verbose=True, factor=0.5)
    dscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Decoderoptim, patience=5, verbose=True, factor=0.5)

    loss_avg = []

    Encoder.train()
    Decoder.train()

    for epoch in range(epoches):
        for sample in dataset_iterator.iterate(4):
            nninput = sample.question
            nnexcept = sample.exceptAnswer

            encoderout, (cx,tx) = Encoder(nninput)
            cx = cx.reshape(Encoder.n_layers, 1, Encoder.hidden_size)
            tx = tx.reshape(Encoder.n_layers, 1, Encoder.hidden_size)
            output, hidden = Decoder(rolt.view(-1, 1), (cx, tx), encoderout)
            outputs = output

            for train in nnexcept[:-1:]:
                output, hidden = Decoder(train.view(-1,1), hidden, encoderout)
                outputs=torch.cat((outputs, output))

            loss = loss_func(outputs.view(-1,Decoder.out_size),nnexcept)
            loss.backward()

            Encoderoptim.step()
            Decoderoptim.step()

            Encoderoptim.zero_grad()
            Decoderoptim.zero_grad()

            loss_avg.append(loss.item())
            if len(loss_avg) >= 50:
                mean_loss = np.mean(loss_avg)
                print(f'Loss: {mean_loss}')

                escheduler.step(mean_loss)
                dscheduler.step(mean_loss)

                loss_avg = []
    torch.save(Encoder,"Encoderdata.pkl")
    torch.save(Decoder,'Decoderdata.pkl')

if __name__ == "__main__":
    import Seq2Seqmodule as S2S

    with open("ai.json") as config:
        parametrs = json.load(config)


    tokens = tk.TokenDictionary.load(".aistate/tokens.json")
    tokenizer = tk.Tokenizer(tokens)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Encoder = S2S.Encoder(**parametrs).to(device)
    Decoder = S2S.Decoder(**parametrs).to(device)

    dataset = ds.load()

    train(20, Encoder, Decoder, device, tokenizer, dataset)
