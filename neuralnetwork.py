"""Provides NN models to train and eval its"""
from __future__ import annotations
from typing import Optional, Self, Tuple
import torch
from torch import nn
import servicetokens as st
import config

class NeuralNetwork:
    """Composite class that represents all neural network"""
    def __init__(self, configuration: config.ConfigurationScope):
        self.__encoder = configuration.resolve(Encoder, "encoder")
        self.__decoder = configuration.resolve(Decoder, "decoder")
        self.__hidden_size = configuration.this()["hidden_size"]
        self.__out_size = configuration.this()["vocabulary_size"] + st.SERVICE_OUTPUT_SIZE

    @property
    def decoder(self) -> Decoder:
        """Decoder part of NN"""
        return self.__decoder

    @property
    def encoder(self) -> Encoder:
        """Encoder part of NN"""
        return self.__encoder

    @property
    def hidden_size(self) -> int:
        """Size of hidden state of NN models"""
        return self.__hidden_size

    @property
    def out_size(self) -> int:
        """Size of output layer of decoder"""
        return self.__out_size

    # pylint: disable=C0103
    def to(self, device: str) -> Self:
        """Sends NN models to target device"""
        self.__decoder = self.decoder.to(device)
        self.__encoder = self.encoder.to(device)
        return self
    # pylint: enable=C0103


class Encoder(nn.Module):
    """Encoder part of NN"""
    def __init__(self,
            vocabulary_size: int,
            embed_size: int,
            hidden_size: int,
            lstm_n_layers: int,
        **_):

        super().__init__()

        self.__embedding = nn.Embedding(vocabulary_size, embed_size)
        self.__lstm = nn.LSTM(embed_size, hidden_size, lstm_n_layers)


    def forward(self, value: torch.LongTensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward propogation of NN"""

        value = self.__embedding(value)
        encoder_out, hidden = self.__lstm(value, hidden)

        return encoder_out, hidden

class Decoder(nn.Module):
    """Decoder part of NN"""
    def __init__(self,
            vocabulary_size: int,
            hidden_size: int,
        **_):

        super().__init__()

        out_size = vocabulary_size + st.SERVICE_OUTPUT_SIZE

        self.__lstm_cell = nn.LSTMCell(hidden_size, hidden_size)
        self.__output_linear = nn.Linear(hidden_size, out_size, bias = False)

        self.__log_softmax = nn.LogSoftmax(dim = -1)


    def forward(self, encoder_out: torch.LongTensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward propogation of NN"""

        out_raw, hidden = self.__lstm_cell(encoder_out, hidden)
        out = self.__output_linear(out_raw)

        out = self.__log_softmax(out)

        return out, (out_raw, hidden)
