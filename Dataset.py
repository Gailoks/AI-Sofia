import os


class QAPair:
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer

class Dialog:
    def __init__(self, dialog:list):
        self.__dialog = dialog

    def listPairs(self) -> list:
        return self.__dialog
    
class Dataset:
    def __init__(self):
        self.__dialogs:list = []
    
    def listDialogs(self) -> list:
        return self.__dialogs
    
    def pushDialog(self, dialog:Dialog) -> None:
        self.__dialogs.append(dialog)


def load() -> Dataset:
    samples = []
    for sample in os.listdir('samples'):
        with open("samples/" + sample, encoding="utf-8") as text:
            samples.append(text.read().lower())

    dataset = Dataset()

    for sample in samples:
        lines = sample.splitlines()
        questions = lines[::2]
        answers = lines[1::2]
        dialog = Dialog(list(map(lambda a: QAPair(a[0], a[1]), zip(questions, answers))))
        dataset.pushDialog(dialog)

    return dataset

if __name__ == "__main__":

    from termcolor import colored

    dataset = load()
    for dialog in dataset.listDialogs():
        for index, pair in enumerate(dialog.listPairs()):
            print(colored(f"[{index}]: ", "blue"), colored(pair.question, "green"), "->", colored(pair.answer, "red"))
        print(colored("■" * 150, "cyan"))
        print("Dialog print end, press any key to show next.")
        input()

