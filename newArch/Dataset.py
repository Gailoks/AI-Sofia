import torch
import Tokens as tk
import os

class Dataset():
    def __init__(self):
        self.qa_pairs:list(Dataset.QAPair) = []
        self.raw_samples:list(str) = []


    def push_sample(self, sample:str) -> None:
        self.raw_samples.append(sample)
        lines = sample.splitlines()
        questions = lines[::2]
        answers = lines[1::2]
        for q, a in zip(questions, answers):
            self.qa_pairs.append(Dataset.QAPair(q, a))


    class QAPair:
        def __init__(self, question, answer):
            self.question = question
            self.answer = answer


class DatasetIterator():
    def __init__(self, device, dataset:Dataset, tokenizer:tk.Tokenizer, **options):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.options = options
        self.device = device
        self.end_token = options.get("end_token", None)
    
    def get_batch(self):
        for qa in self.dataset.qa_pairs:
            question_idx = list(self.tokenizer.tokenize(qa.question))

            target = list(self.tokenizer.tokenize(qa.answer))

            if self.end_token != None:
                target.append(self.end_token)
                        
            test = question_idx + target[:-1]

            target = torch.LongTensor(target).to(self.device)
            test = torch.LongTensor(test).to(self.device)
            yield target, test


class DatasetLoader():
    def __init__(self, **options):
        self.options = options

    def load(self, path:str) -> Dataset:
        dataset = Dataset()
        for sample in os.listdir(path):
            with open(f"{path}/{sample}", encoding="utf-8") as text:
                dataset.push_sample(text.read().lower())
        return dataset