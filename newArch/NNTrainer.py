import torch
import Tokens as tk
from AIState import *
from Dataset import *
import numpy as np

class NNTrainer():
    def __init__(self, state:AIState, dataset:DatasetIterator, **options):
        self.options = options
        self.state = state
        self.dataset = dataset
        self.batch_size = options.get("batch_size", 1)


    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.state.module.parameters(), lr=1e-2, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            patience=5, 
            verbose=True, 
            factor=0.5
        )

        loss_avg = []
        epoch = 0

        while True:
            epoch += 1

            for target, train in self.dataset.get_batch():
                self.state.module.train()

                hidden = self.state.module.create_empty_hidden(self.batch_size)

                output, hidden = self.state.module(train, hidden)
                target_len = len(target)
                loss = criterion(output[-target_len:], target)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss_avg.append(loss.item())

                if len(loss_avg) >= 50:
                    mean_loss = np.mean(loss_avg)
                    scheduler.step(mean_loss)
                    self.eval()
                    loss_avg = []

                    yield NNTrainer.EpochResult(epoch, mean_loss)


    class EpochResult:
        def __init__(self, epoch, mean_loss):
            self.epoch = epoch
            self.mean_loss = mean_loss