from typing import Iterable
import Dataset as ds
import ServiceTokens as st
import Tokens as tk
from LernSample import *


class DatasetIterator:
    def __init__(self, dataset:ds.Dataset, servicetokens:st.ServiceTokens, tokenizer:tk.Tokenizer, device):
        self.__dataset = dataset
        self.__servicetokens = servicetokens
        self.__device = device
        self.__tokenizer = tokenizer

    def iterate(self) -> Iterable(LernSample):
        for qapair in self.__dataset.listPairs():
            q = torch.Tensor(self.__tokenizer.tokenize(qapair.question) + [self.__servicetokens.get(st.STI_ROLE)]).to(self.__device)
            a = torch.Tensor(self.__tokenizer.tokenize(qapair.answer) + [self.__servicetokens.get(st.STO_END)]).to(self.__device)

            yield LernSample(q, a)