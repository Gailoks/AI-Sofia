from typing import Iterable
import Dataset as ds
import ServiceTokens as st
import Tokens as tk
from Batch import *
import random
import torch


class DatasetIterator:
    def __init__(self, dataset:ds.Dataset, servicetokens:st.ServiceTokens, tokenizer:tk.Tokenizer):
        self.__dataset = dataset
        self.__servicetokens = servicetokens
        self.__tokenizer = tokenizer

    def iterate(self, batch_size, out_len) -> Iterable:
        qaPairs = self.__dataset.listPairs()
        
        samples: list = []

        for qa in qaPairs:
            tokenized_question = list(self.__tokenizer.tokenize(qa.question))

            question_tensor = torch.LongTensor(tokenized_question)

            tokenized_answer = list(self.__tokenizer.tokenize(qa.answer + " "))
            tokenized_answer = tokenized_answer + [self.__servicetokens.get(st.STIO_NULL)] * (out_len - len(tokenized_answer))

            answer_tensor = torch.LongTensor(tokenized_answer).view(-1, 1) # Transform to 1-d vertical column tensor

            samples.append(LearnSample(question_tensor, answer_tensor))

        offset = 0
        for offset in range(0, len(samples), batch_size):
            raw_batch = samples[offset:offset + batch_size:]

            batch = Batch(out_len)
            for batch_el in raw_batch:
                batch.append(batch_el)

            yield batch
            
if __name__ == "__main__":
    from termcolor import colored

    dataset = ds.load()

    tokens = tk.TokenDictionary.load(".aistate/tokens.json")
    tokenizer = tk.Tokenizer(tokens)

    service_tokens = st.ServiceTokens(tokenizer.count_tokens())

    iterator = DatasetIterator(dataset, service_tokens, tokenizer)

    for batch in iterator.iterate(3, 100):
        for qa in zip(batch.questions, batch.answers.permute(1, 0)):
            print(f"{colored(qa[0], 'green')} -> {colored(qa[1], 'red')}")
        input()
        
