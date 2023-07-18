"""Trainer for NN"""
# pylint: disable=E1101
# pylint: disable=R0903
from __future__ import annotations
import random
from typing import Iterable, Optional
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import texts
import tokens
from neuralnetwork import NeuralNetwork
import servicetokens as st
from batching import Batch, LearnSample

class _DatasetIterator:
    """Splits dataset to batches to train on it"""
    def __init__(self,
            dataset: texts.Dataset,
            tokenizer: tokens.Tokenizer
        ):

        self.__dataset = dataset
        self.__tokenizer = tokenizer

    def iterate(self, batch_size: int) -> Iterable[Batch]:
        """Iterates dataset and creates batches of target size"""
        pairs = self.__dataset.list_pairs()

        samples: list[LearnSample] = []

        for pair in pairs:
            tokenized_question = list(self.__tokenizer.tokenize(pair.question))
            question_tensor = torch.LongTensor(tokenized_question)

            tokenized_answer = list(self.__tokenizer.tokenize(pair.answer))
            answer_tensor = torch.LongTensor(tokenized_answer)

            samples.append(LearnSample(pair.question, pair.answer, question_tensor, answer_tensor))

        offset = 0
        for offset in range(0, len(samples), batch_size):
            raw_batch = samples[offset:offset + batch_size:]

            batch = Batch()
            for batch_el in raw_batch:
                batch.append(batch_el)

            yield batch

# pylint: disable=C0116
class _TrainEnvironment:
    class _ModelEnvironment:
        def __init__(self, optimizer: torch.optim.Optimizer, scheduler_kwargs: dict[str, any]):
            self.optimizer = optimizer
            self.scheduler = ReduceLROnPlateau(optimizer, **scheduler_kwargs)


    def __init__(self,
            loss_func: nn.NLLLoss,
            encoder_optimizer: torch.optim.Optimizer,
            decoder_optimizer: torch.optim.Optimizer,
            scheduler_kwargs: dict[str, any],
            scheduler_step_size: int
        ):

        self.__encoder_te = _TrainEnvironment._ModelEnvironment(encoder_optimizer, scheduler_kwargs)
        self.__decoder_te = _TrainEnvironment._ModelEnvironment(decoder_optimizer, scheduler_kwargs)
        self.__loss_avg = []
        self.__loss_func = loss_func
        self.__scheduler_step_size = scheduler_step_size

    def loss(self, real: torch.Tensor, excpted: torch.LongTensor):
        return self.__loss_func(real, excpted)

    def apply_loss(self, loss) -> None:
        loss.backward()

        self.__encoder_te.optimizer.step()
        self.__decoder_te.optimizer.step()

        self.__encoder_te.optimizer.zero_grad()
        self.__decoder_te.optimizer.zero_grad()

        self.__loss_avg.append(loss.item())

    def scheduler_step_ifneed(self) -> Optional[float]:
        if len(self.__loss_avg) >= self.__scheduler_step_size:
            mean_loss = np.mean(self.__loss_avg)

            self.__encoder_te.scheduler.step(mean_loss)
            self.__decoder_te.scheduler.step(mean_loss)

            self.__loss_avg = []

            return mean_loss
        return None
# pylint: enable=C0116

# pylint: disable=R0913
# pylint: disable=R0914
class Trainer:
    """Main component of NN traning"""
    def __init__(self,
            batch_size: int,
            learn_rate: float,
            scheduler_patience: int,
            scheduler_factor: float,
            scheduler_step_size: int,
        **_):

        self.__batch_size = batch_size
        self.__learn_rate = learn_rate
        self.__scheduler_step_size = scheduler_step_size
        self.__scheduler_kwargs = {
            "patience": scheduler_patience,
            "factor": scheduler_factor,
            "verbose": True
        }


    def train(self,
            network: NeuralNetwork,
            tokenizer: tokens.Tokenizer,
            dataset: texts.Dataset,
            epoch_count: int,
            device: str
        ):
        """Trains NN model with given parameters on dataset"""

        network.to(device)

        service_tokens = st.ServiceTokens(tokenizer.count_tokens())
        dataset_iterator = _DatasetIterator(dataset, tokenizer)

        network.encoder.train()
        network.decoder.train()

        train_environment = _TrainEnvironment(
            nn.NLLLoss(),
            torch.optim.Adamax(network.encoder.parameters(), lr = self.__learn_rate),
            torch.optim.Adam(network.decoder.parameters(), lr = self.__learn_rate, amsgrad=True),
            self.__scheduler_kwargs,
            self.__scheduler_step_size
        )


        for epoch in range(epoch_count):
            for batch in dataset_iterator.iterate(self.__batch_size):

                batch.to(device)
                samples = batch.list_samples()
                random.shuffle(samples)

                null_input = torch.zeros(network.hidden_size).to(device)

                encoder_hidden = None
                decoder_hidden = None

                sto_switch = torch.LongTensor([service_tokens.get(st.STO_SWITCH)]).to(device)

                decoder_errors = [
                    torch.Tensor().view(0, network.out_size).to(device),
                    torch.LongTensor().view(0).to(device)
                ]

                def add_mistake(real:torch.Tensor, excepted:torch.LongTensor):
                    real = real.view(1, network.out_size)
                    excepted = excepted.view(1)
                    decoder_errors[0] = torch.cat((decoder_errors[0], real), dim=0)
                    decoder_errors[1] = torch.cat((decoder_errors[1], excepted), dim=0)

                for sample in samples:

                    # Run encoder
                    encoder_outs, encoder_hidden = network.encoder(sample.question, encoder_hidden)

                    #Phase I (excepted any non STO_SWITCH)
                    for encoder_out in encoder_outs[:-1]:
                        decoder_out, decoder_hidden = network.decoder(encoder_out, decoder_hidden)
                        if decoder_out.argmax() == sto_switch:
                            random_value = random.randint(0, tokenizer.count_tokens() - 1)
                            random_tensor = torch.LongTensor([random_value]).to(device)
                            add_mistake(decoder_out, random_tensor)
                            add_mistake(decoder_out, random_tensor)

                    # Phase II (excepted STO_SWITCH)
                    decoder_out, decoder_hidden = network.decoder(encoder_outs[-1], decoder_hidden)
                    add_mistake(decoder_out, sto_switch)

                    # Phase III (excepted answer)
                    for answer_peace in sample.answer:
                        decoder_out, decoder_hidden = network.decoder(null_input, decoder_hidden)
                        add_mistake(decoder_out, answer_peace)

                    # Phase IV (excepted STO_SWITCH)
                    decoder_out, decoder_hidden = network.decoder(null_input, decoder_hidden)
                    add_mistake(decoder_out, sto_switch)


                loss = train_environment.loss(*decoder_errors)

                train_environment.apply_loss(loss)

                print(f"Trainer optimizer step: [epoch: {epoch}/{epoch_count}, loss: {loss}]")

                mean_loss_value = train_environment.scheduler_step_ifneed()
                if mean_loss_value is not None:
                    print(f"Trainer scheduler correction: [epoch: {epoch}/{epoch_count}, mean loss: {mean_loss_value}]")
# pylint: enable=R0913
# pylint: enable=R0914
