import torch

class LearnSample:
    def __init__(self, question:torch.LongTensor, exceptAnswer:torch.LongTensor):
        self.question = question
        self.exceptAnswer = exceptAnswer

