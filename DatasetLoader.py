import os


class QAPair:
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer


def load() -> list():
    samples = []
    for sample in os.listdir('samples'):
        with open("samples/" + sample, encoding="utf-8") as text:
            samples.append(text.read().lower())

    dataset = []

    for sample in samples:
        lines = sample.splitlines()
        questions = lines[::2]
        answers = lines[1::2]
        for q, a in zip(questions, answers):
            dataset.append(QAPair(q, a))
    return dataset
