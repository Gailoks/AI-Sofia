import torch.nn as nn
import torch
import numpy as np
import json
import Dataset as ds
import Tokens as tk
import ServiceTokens as st
import DatasetIterator as dsi
import Seq2Seqmodule as s2s


class TrainEnvironment:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True, factor=0.5)

class Trainer:
    def __init__(self, loss_func, decoder_te: TrainEnvironment, encoder_te: TrainEnvironment):
        self.encoder_te = encoder_te
        self.decoder_te = decoder_te
        self.loss_avg = []
        self.loss_func = loss_func
        pass

    def optim_step(self):
        self.encoder_te.optimizer.step()
        self.decoder_te.optimizer.step()

        self.encoder_te.optimizer.zero_grad()
        self.decoder_te.optimizer.zero_grad()

    def add_loss(self, loss):
        self.loss_avg.append(loss.item())

    def scheduler_step(self):
        mean_loss = np.mean(loss_avg)
        print(f'Loss: {mean_loss}')

        self.encoder_te.scheduler.step(mean_loss)
        self.decoder_te.scheduler.step(mean_loss)

        loss_avg = []



def train(epoches: int, encoder: s2s.Encoder, decoder: s2s.Decoder, tokenizer: tk.Tokenizer, dataset: ds.Dataset, device: str, batch_size: int = 20, out_len: int = 100) -> None:

    service_tokens = st.ServiceTokens(tokenizer.count_tokens())
    dataset_iterator = dsi.DatasetIterator(dataset, service_tokens, tokenizer)

    loss_func = nn.NLLLoss()
    encoder_te = TrainEnvironment(torch.optim.Adam(encoder.parameters(), lr = 1e-2))
    decoder_te = TrainEnvironment(torch.optim.Adam(decoder.parameters(), lr = 1e-2))

    trainer = Trainer(loss_func, encoder_te, decoder_te)

    encoder.train()
    decoder.train()

    for epoch in range(epoches):
        
        for batch in dataset_iterator.iterate(batch_size, out_len):

            batch.to(device)

            real_batch_size = batch.size()

            hiddens = torch.LongTensor().to(device)
            for question in batch.questions:
                x, hidden = encoder(question)
                # For example: 2, 700 -> 2, 1, 700. It will joined in 3-d by second dim and in result we have 2, real_batch_size, 700
                hidden = hidden.view(encoder.n_layers, 1, encoder.hidden_size)
                hiddens = torch.cat((hiddens, hidden), 1)

            decoder_input = torch.LongTensor([service_tokens.get(st.STIO_NULL)] * real_batch_size).view(-1, 1).to(device) #Init with null value

            decoder_output, hiddens = decoder(decoder_input, hiddens)

            print(decoder_output)


            for sample in dataset_iterator.iterate(25, 150):
                nninput = sample.question
                nnexcept = sample.exceptAnswer

                encoderout, (cx,tx) = encoder(nninput)
                cx = cx.reshape(encoder.n_layers, 1, encoder.hidden_size)
                tx = tx.reshape(encoder.n_layers, 1, encoder.hidden_size)
                output, hidden = decoder(rolt.view(-1, 1), (cx, tx), encoderout)
                outputs = output

                for train in nnexcept[:-1:]:
                    output, hidden = decoder(train.view(-1,1), hidden, encoderout)
                    outputs=torch.cat((outputs, output))

                loss = loss_func(outputs.view(-1,decoder.out_size),nnexcept)
                loss.backward()

                trainer.optim_step()



if __name__ == "__main__":
    with open("ai.json") as config:
        parametrs = json.load(config)


    tokens = tk.TokenDictionary.load(".aistate/tokens.json")
    tokenizer = tk.Tokenizer(tokens)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = s2s.Encoder(**parametrs).to(device)
    decoder = s2s.Decoder(**parametrs).to(device)

    dataset = ds.load()

    train(20, encoder, decoder, tokenizer, dataset, device)

    torch.save(encoder, ".aistate/encoder.pkl")
    torch.save(decoder, ".aistate/decoder.pkl")


def batching_hidden(hiddens, batch_size):
    cx, tx = hiddens[0]
    num_layers, hidden_size = cx.shape

    for hidden in hiddens[1:]:
        cx1, tx1 = hidden
        cx = torch.cat((cx,cx1))
        tx = torch.cat((tx, tx1))
    cx = cx.reshape(num_layers, batch_size, hidden_size)
    tx = tx.reshape(num_layers, batch_size, hidden_size)
    return (cx, tx)