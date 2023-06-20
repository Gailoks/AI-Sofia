from typing import Iterable
import Dataset as ds
import ServiceTokens as st
import Tokens as tk
from LearnSample import *
import random
import torch


class DatasetIterator:
    def __init__(self, dataset:ds.Dataset, servicetokens:st.ServiceTokens, tokenizer:tk.Tokenizer, device):
        self.__dataset = dataset
        self.__servicetokens = servicetokens
        self.__device = device
        self.__tokenizer = tokenizer

    def iterate(self, combine_question:int) -> Iterable:

        qaPairs = DatasetIterator.__combinations(self.__dataset.listPairs(), combine_question)
        
        for qapairComb in qaPairs:

            superQ = []
            superA = []

            for qaPair in qapairComb:
                superQ += list(self.__tokenizer.tokenize(qaPair.question))
                superA += list(self.__tokenizer.tokenize(qaPair.answer))

            superQ.append(self.__servicetokens.get(st.STI_ROLE))
            superA.append(self.__servicetokens.get(st.STO_END))

            q = torch.LongTensor(superQ).to(self.__device)
            a = torch.LongTensor(superA).to(self.__device)

            yield LearnSample(q, a)

    @staticmethod
    def __combinations(iterable, r):
        # combinations('ABCD', 2) --> AB AC AD BC BD CD
        # combinations(range(4), 3) --> 012 013 023 123
        pool = tuple(iterable)
        n = len(pool)
        if r > n:
            return
        indices = list(range(r))
        yield tuple(pool[i] for i in indices)
        while True:
            for i in reversed(range(r)):
                if indices[i] != i + n - r:
                    break
            else:
                return
            indices[i] += 1
            for j in range(i+1, r):
                indices[j] = indices[j-1] + 1
            yield tuple(pool[i] for i in indices)


if __name__ == "__main__":
    from termcolor import colored

    dataset = ds.load()

    tokens = tk.TokenDictionary.load(".aistate/tokens.json")
    tokenizer = tk.Tokenizer(tokens)

    service_tokens = st.ServiceTokens(tokenizer.count_tokens())

    iterator = DatasetIterator(dataset, service_tokens, tokenizer, torch.device("cpu"))

    for sample in iterator.iterate(2):
        print(f"{colored(sample.question, 'green')} -> {colored(sample.exceptAnswer, 'red')}", end=None)
        input()
        