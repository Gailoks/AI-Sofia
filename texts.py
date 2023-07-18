"""Module to work and load texts samples"""
import os
from dataclasses import dataclass


class TextsDatabase:
    """Primitive database that stores raw data from files"""

    def __init__(self):
        self.__texts: list[str] = []

    def add_text(self, text: str) -> None:
        """Adds new text to database"""
        self.__texts.append(text)

    def list_texts(self) -> list[str]:
        """Lists all texts from database"""
        return self.__texts


def load_text_database(path: str) -> TextsDatabase:
    """Loads database from files in given directory"""
    database = TextsDatabase()
    for sample in os.listdir(path):
        with open(path + "/" + sample, encoding="utf-8") as text:
            database.add_text(text.read())

    return database


@dataclass
class QAPair:
    """Represents pair of question and answer for ai"""

    question: str
    answer: str


class Dataset:
    """Dataset for NN learning"""

    def __init__(self):
        self.__pairs: list = []

    def list_pairs(self) -> list[QAPair]:
        """Lists all QAPairs form dataset"""
        return self.__pairs

    def push_pair(self, pair: QAPair) -> None:
        """Adds new QAPairs to dataset"""
        self.__pairs.append(pair)


def load_dataset(texts_database: TextsDatabase) -> Dataset:
    """Creates new dataset from TextsDatabase"""
    seporator = "\n\n"

    dataset = Dataset()

    for text in texts_database.list_texts():
        for i, pair in enumerate(text.split(seporator)):
            lines = pair.splitlines()
            if len(lines) != 2:
                raise RuntimeError(
                    f'''Dataset load error, input dataset has invalid format,
                        record at line {i * 3 + 1} has invalid count of lines'''
                )

            qa_pair = QAPair(lines[0], lines[1])

            dataset.push_pair(qa_pair)

    return dataset
