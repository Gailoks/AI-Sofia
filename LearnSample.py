import torch

class LearnSample:
    def __init__(self, question:torch.LongTensor, exceptAnswer:torch.LongTensor, mask:torch.BoolTensor):
        self.question = question
        self.exceptAnswer = exceptAnswer
        self.mask = mask 
