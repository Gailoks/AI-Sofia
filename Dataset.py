import os


class QAPair:
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer

class Dataset:
    def __init__(self):
        self.__pairs: list = []

    def listPairs(self) -> list:
        return self.__pairs

    def pushPair(self, pair:QAPair) -> None:
        self.__pairs.append(pair)


def load(path:str = "samples") -> Dataset:
    samples = []
    for sample in os.listdir(path):
        with open(path + "/" + sample, encoding="utf-8") as text:
            samples.append(text.read())

    dataset = Dataset()

    seporator = "\n\n"

    for sample in samples:
        for pair in sample.split(seporator):
            lines = pair.splitlines()
            q = lines[0]
            a = lines[1]
            dataset.pushPair(QAPair(q, a))

    return dataset


if __name__ == "__main__":
    from termcolor import colored

    dataset = load()
    for index, pair in enumerate(dataset.listPairs()):
        print(colored(f"[{index}]: ", "blue"), colored(
            pair.question, "green"), "->", colored(pair.answer, "red"))
