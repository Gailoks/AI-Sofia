import os


class QAPair:
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer


class Dialog:
    def __init__(self, dialog: list):
        self.__dialog = dialog

    def listPairs(self) -> list:
        return self.__dialog


class Dataset:
    def __init__(self):
        self.__dialogs: list = []

    def listDialogs(self) -> list:
        return self.__dialogs

    def pushDialog(self, dialog: Dialog) -> None:
        self.__dialogs.append(dialog)


def load(path:str = "samples") -> Dataset:
    samples = []
    for sample in os.listdir(path):
        with open(path + "/" + sample, encoding="utf-8") as text:
            samples.append(text.read())

    dataset = Dataset()

    seporator = "\nEND_DIALOG\n"

    for sample in samples:
        for dialog in sample.split(seporator):
            lines = dialog.splitlines()
            questions = lines[::2]
            answers = lines[1::2]
            dialog = Dialog(
                list(map(lambda a: QAPair(a[0].lower(), a[1].lower()), zip(questions, answers))))
            dataset.pushDialog(dialog)

    return dataset


if __name__ == "__main__":
    from termcolor import colored

    dataset = load()
    for dialog in dataset.listDialogs():
        for index, pair in enumerate(dialog.listPairs()):
            print(colored(f"[{index}]: ", "blue"), colored(
                pair.question, "green"), "->", colored(pair.answer, "red"))
        print(colored("■" * 150, "cyan"))
        print("Dialog print end, press any key to show next.")
        input()
