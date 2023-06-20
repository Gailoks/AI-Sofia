import torch

class LernSample:
    def __init__(self, question:torch.Tensor, exceptAnswer:torch.Tensor):
        self.question = question
        self.exceptAnswer = exceptAnswer
