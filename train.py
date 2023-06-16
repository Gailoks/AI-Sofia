import torch.nn as nn
import torch
import numpy as np
import json
import Dataset as ds
import Tokens as tk
import ServiceTokens as st


def train(epoches: int, Encoder:nn.Module, Decoder:nn.Module, device: str, tokenizer, dataset) -> None:
    """epoches - number of epoches through all dataset
    model - model required to teach
    batch_size - n/a"""
    service_tokens = st.ServiceTokens(tokenizer.count_tokens())

    loss_func = nn.NLLLoss()

    Encoderoptim = torch.optim.Adamax(Encoder.parameters(), lr=1e-2)
    Decoderoptim =torch.optim.Adam(Decoder.parameters(), lr=1e-2)

    escheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Encoderoptim, patience=5, verbose=True, factor=0.5)
    dscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Decoderoptim, patience=5, verbose=True, factor=0.5)
    
    def get_batch(dialog: ds.Dialog):
        nninput = []
        for qa in dialog.listPairs():
            nninput += list(tokenizer.tokenize(qa.question)) + [service_tokens.get(st.STI_ROLE)]
            nnexcept = list(tokenizer.tokenize(qa.answer)) + [service_tokens.get(st.STO_END)]
            yield nninput.copy(), nnexcept.copy()
            nninput += list(tokenizer.tokenize(qa.answer)) + [service_tokens.get(st.STI_ROLE)]

    loss_avg = []

    Encoder.train()
    Decoder.train()

    for epoch in range(epoches):

        for dialog in dataset.listDialogs():

            for nninput, nnexcept in get_batch(dialog):

                encoderout, hidden = Encoder(torch.LongTensor(nninput).to(device))
                output, hidden = Decoder(torch.LongTensor([nninput[-1]]).to(device), hidden, encoderout)
                outputs = output

                for train in nnexcept[:-1:]:
                    output, hidden = Decoder(torch.LongTensor([train]).to(device), hidden, encoderout)
                    outputs=torch.cat((outputs, output))

                loss = loss_func(outputs.view(-1,Decoder.out_size),torch.LongTensor(nnexcept).to(device))
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
