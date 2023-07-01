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
        """
        batch_size
        batch_len
        """

        qaPairs =self.__dataset.listPairs()
        
        train = []
        target = [] 
        

        for sample in qaPairs:
            tq = list(self.__tokenizer.tokenize(sample.question))#tokenized question
            ta = list(self.__tokenizer.tokenize(sample.answer+" "))#tokenized answer
            train += tq
            target += ta + [self.__servicetokens.get(st.STIO_NULL)]*(out_size - len(ta))
            

        
        offset = 0
        for y in range(0,len(train)-batch_size,batch_size):
                inptensor = torch.LongTensor(train[offset:batch_size+offset]).to(self.__device)
                outtensor = torch.LongTensor(target[offset:batch_size+offset]).to(self.__device)
                offset+= batch_size
                yield LearnSample(inptensor, outtensor)
        self.__dataset.shuffle()


if __name__ == "__main__":
    from termcolor import colored

    dataset = ds.load()

    tokens = tk.TokenDictionary.load(".aistate/tokens.json")
    tokenizer = tk.Tokenizer(tokens)

    service_tokens = st.ServiceTokens(tokenizer.count_tokens())

    iterator = DatasetIterator(dataset, service_tokens, tokenizer, torch.device("cpu"))

    for sample in iterator.iterate(1,5):
        print(f"{colored(sample.question, 'green')} -> {colored(sample.exceptAnswer, 'red')}", end=None)
        input()
        
