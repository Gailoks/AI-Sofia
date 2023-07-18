"""Provides functional to batch dataset"""
# pylint: disable=E1101
from dataclasses import dataclass
import torch

@dataclass
class LearnSample:
    """One sample for NN learing"""
    question_text: str
    answer_text: str

    question: torch.LongTensor
    answer: torch.LongTensor

class Batch:
    """Represents on batch for NN learning"""
    def __init__(self):
        self.samples: list[LearnSample] = []

    def append(self, sample: LearnSample) -> None:
        """Adds new learn sample to batch"""
        self.samples.append(sample)

    def list_samples(self) -> list[LearnSample]:
        """Lists all samples of batch"""
        return self.samples

    # pylint: disable=C0103
    def to(self, device):
        """Sends all batch to given device"""
        for sample in self.samples:
            sample.question = sample.question.to(device)
            sample.answer = sample.answer.to(device)
    # pylint: enable=C0103

    def size(self):
        """Returns real size of batch"""
        return len(self.samples)
    