"""Provides functional to tokenize text"""
from __future__ import annotations
import json
from collections import Counter
from pymorphy3 import MorphAnalyzer
import texts


VOCABULARY_TOKENS_COUNT = 1
VT_UNKOWN = 0


def _is_valid_char(char: str) -> bool:
    return str.isalpha(char) or char in [' ', '\n']

def _preprocess_text(text: str):
    text = text.lower()
    text = "".join(filter(_is_valid_char, text))
    return text


def _split_text(text: str, morph: MorphAnalyzer):
    def normalize_word(word: str):
        parse_result = morph.parse(word)
        most_scored = max(parse_result, key=lambda x: x.score)
        return most_scored.normalized.word

    return list(map(normalize_word, _preprocess_text(text).split()))


class TokenDictionary:
    """Represents dictionary of tokens"""

    def __init__(self, tokens_decode_table: list[str]):
        self.tokens = tokens_decode_table
        self.rw_tokens = {v: k for k, v in enumerate(self.tokens)}

    def code(self, text:str) -> int:
        """Tries to code text to token, if failed returns VT_UNKOWN"""
        for token, token_text in enumerate(self.tokens):
            if text == token_text:
                return token + VOCABULARY_TOKENS_COUNT

        return VT_UNKOWN

    def decode(self, token: int) -> str:
        """Decodes token to text that associated with it, if VT_UNKOWN raises excpetion"""
        if token == VT_UNKOWN:
            raise KeyError("Enable to decode VT_UNKOWN token")
        return self.tokens[token - VOCABULARY_TOKENS_COUNT]

    def count(self) -> int:
        """Counts tokens in dictionary"""
        return len(self.tokens)

    def save(self, path: str) -> None:
        """Saves dictionary to file"""
        with open(path, "w", encoding="utf-16") as file:
            json.dump(self.tokens, file, ensure_ascii=False)

    @staticmethod
    def load(path: str) -> TokenDictionary:
        """Loads dictionary from file"""
        with open(path, "r", encoding="utf-16") as file:
            tokens = json.load(file)
            return TokenDictionary(tokens)

    def __str__(self) -> str:
        return "TokenDictionary" + str(self.tokens)


class TokenDictionaryGenerator:
    """Generator of token dictionary"""
    def __init__(self, vocabulary_size: int):
        self.__vocabulary_size_fixed = vocabulary_size - VOCABULARY_TOKENS_COUNT

    def generate_tokens(self, database: texts.TextsDatabase) -> TokenDictionary:
        """Analyzes given TextDatabase and builds optimal tokens model"""

        tokens = [",", ".", "-", "?"]

        data = []

        morph = MorphAnalyzer()

        for text in database.list_texts():
            data += _split_text(text, morph)

        counter = Counter(data)
        tokens += list(
            map(
                lambda x: x[0],
                counter.most_common(self.__vocabulary_size_fixed - len(tokens)),
            )
        )

        return TokenDictionary(tokens)


class Tokenizer:
    """Token processor, tokenizes and detokenizes text using TokenDictionary"""
    def __init__(self, dictionary: TokenDictionary):
        self.__dictionary = dictionary
        self.__morph = MorphAnalyzer()

    def tokenize(self, text: str) -> list[int]:
        """Creates list of token that represents your text"""
        result = []

        for word in _split_text(text, self.__morph):
            token = self.__dictionary.code(word)

            result.append(token)

        return result

    def detokenize(self, tokens: list[int]) -> str:
        """Converts token list to text"""
        result = []

        for token in tokens:
            if token == VT_UNKOWN:
                word = "UNKNOWN"
            word = self.__dictionary.decode(token)
            result.append(word)

        return " ".join(result)

    def count_tokens(self) -> int:
        """Counts tokens in base TokenDictionary"""
        return self.__dictionary.count()

    def get_dictionary(self) -> TokenDictionary:
        """Reruens base TokenDictionary"""
        return self.__dictionary
    