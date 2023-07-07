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

    def scheduler_step(self,loss_avg):
        mean_loss = np.mean(loss_avg)
        print(f'Loss: {mean_loss}')

        self.encoder_te.scheduler.step(mean_loss)
        self.decoder_te.scheduler.step(mean_loss)



def batching_hidden(hiddens, batch_size=0):
    """batch_size is optional"""
    if not batch_size:
        batch_size = len(hiddens)
    cx, tx = hiddens[0]
    num_layers, batch, hidden_size = cx.shape

    for hidden in hiddens[1:]:
        cx1, tx1 = hidden
        cx = torch.cat((cx,cx1))
        tx = torch.cat((tx, tx1))
    cx = cx.reshape(num_layers, batch_size, hidden_size)
    tx = tx.reshape(num_layers, batch_size, hidden_size)
    return (cx, tx)



def train(epoches: int, encoder: s2s.Encoder, decoder: s2s.Decoder, tokenizer: tk.Tokenizer, dataset: ds.Dataset, device: str, batch_size: int = 20, out_len: int = 100) -> None:

    service_tokens = st.ServiceTokens(tokenizer.count_tokens())
    dataset_iterator = dsi.DatasetIterator(dataset, service_tokens, tokenizer)

    loss_func = nn.NLLLoss()
    encoder_te = TrainEnvironment(torch.optim.Adamax(encoder.parameters(), lr = 1e-2))
    decoder_te = TrainEnvironment(torch.optim.Adam(decoder.parameters(), lr = 1e-2, amsgrad=True))

    trainer = Trainer(loss_func, encoder_te, decoder_te)

    encoder.train()
    decoder.train()

    loss_avg = []

    for epoch in range(epoches):
        
        for batch in dataset_iterator.iterate(batch_size, out_len):

            batch.to(device)

            real_batch_size = batch.size()

            hiddens = []
            for question in batch.questions:
                hidden = encoder(question.view(-1,1))
                hiddens.append(hidden)
            hidden = batching_hidden(hiddens)

            decoder_input = torch.LongTensor([service_tokens.get(st.STIO_NULL)] * real_batch_size).view(1, -1).to(device) #Init with null value

            decoder_outputs, hidden = decoder(decoder_input, hidden)
            
            decoder_output, hidden = decoder(batch.answers[:-1],hidden)
            decoder_outputs = torch.cat((decoder_outputs, decoder_output))

            loss = loss_func(decoder_outputs.view(-1,decoder.out_size), batch.answers.view(-1))
            loss.backward()
           
            trainer.optim_step()

            loss_avg.append(loss.item())
            if len(loss_avg) >= 15:
                trainer.scheduler_step(loss_avg)

                loss_avg = []

if __name__ == "__main__":
    with open("ai.json") as config:
        parametrs = json.load(config)


    tokens = tk.TokenDictionary.load(".aistate/tokens.json")
    tokenizer = tk.Tokenizer(tokens)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = s2s.Encoder(**parametrs).to(device)
    decoder = s2s.Decoder(**parametrs).to(device)

    dataset = ds.load()

    train(40, encoder, decoder, tokenizer, dataset, device,batch_size=5,out_len=67)

    torch.save(encoder, ".aistate/encoder.pkl")
    torch.save(decoder, ".aistate/decoder.pkl")
