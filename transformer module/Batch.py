import torch
import Dataset as ds

class LearnSample:
    def __init__(self, question:torch.LongTensor, exceptAnswer:torch.LongTensor):
        self.question = question
        self.exceptAnswer = exceptAnswer

class Batch:
    def __init__(self, out_len: int):
        self.questions = []
        self.answers = torch.LongTensor()
        self.out_len = out_len
        pass


    def to(self, device):
        self.answers = self.answers.to(device)
        self.questions = list(map(lambda x: x.to(device), self.questions))

    def append(self, sample: LearnSample):
        target_size = torch.Size([self.out_len, 1])
        if sample.exceptAnswer.size() != target_size:
            raise Exception(f"Invalid learn sample size, was {sample.exceptAnswer.size()} except {target_size}")

        self.questions.append(sample.question)
        self.answers = torch.cat((self.answers, sample.exceptAnswer), 1)

    def size(self) -> int:
        return len(self.questions)