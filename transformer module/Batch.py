import torch
import Dataset as ds

class LearnSample:
    def __init__(self, question:torch.LongTensor, exceptAnswer:torch.LongTensor):
        self.question = question
        self.exceptAnswer = exceptAnswer

class Batch:
    def __init__(self) -> None:
        self.samples = []

    def append(self, sample:LearnSample):
        self.samples.append(sample)

    def list_samples(self):
        return self.samples
    
    def to(self, device):
        for sample in self.samples:
            sample.question = sample.question.to(device)
            sample.exceptAnswer = sample.exceptAnswer.to(device)

    def size(self):
        return len(self.samples)