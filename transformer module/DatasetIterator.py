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

    def iterate(self, batch_size, out_size) -> Iterable:
        qaPairs = self.__dataset.listPairs()
        
        samples : list = []

        for qa in qaPairs:
            tokenized_question = list(self.__tokenizer.tokenize(qa.question))

            tokenized_answer = list(self.__tokenizer.tokenize(qa.answer + " "))
            tokenized_answer = tokenized_answer + [self.__servicetokens.get(st.STIO_NULL)] * (out_size - len(tokenized_answer))

            samples.append((tokenized_question, tokenized_answer))

        offset = 0
        for offset in range(0, len(samples), batch_size):
            raw_batch = samples[offset:offset + batch_size:]

            batch = []
            for batch_el in raw_batch:
                batch.append(LearnSample(torch.LongTensor(batch_el[0]).to(self.__device), torch.LongTensor(batch_el[1]).to(self.__device)))

            yield batch
            
if __name__ == "__main__":
    from termcolor import colored

    dataset = ds.load()

    tokens = tk.TokenDictionary.load(".aistate/tokens.json")
    tokenizer = tk.Tokenizer(tokens)

    service_tokens = st.ServiceTokens(tokenizer.count_tokens())

    iterator = DatasetIterator(dataset, service_tokens, tokenizer, torch.device("cpu"))

    for batch in iterator.iterate(3, 100):
        for qa in batch:
            print(f"{colored(qa.question, 'green')} -> {colored(qa.exceptAnswer, 'red')}")
        input()
        
