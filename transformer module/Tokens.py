import json
import Dataset as ds
from collections import Counter
from pymorphy3 import MorphAnalyzer
# TokenDictionary, TokenDictionaryGenerator and Tokenizer


VOCABULARY_TOKENS_COUNT = 1
VT_UNKOWN = 0

def preprocess_text(text: str):
    text = text.lower()
    text = "".join(filter(lambda x: x not in ",.-:\'\"@#$%^&*()_+=/*+?!", text))
    return text

def split_text(text: str, morph: MorphAnalyzer):
    def normalize_word(x):
        parse_result = morph.parse(x)
        most_scored = max(parse_result, key=lambda x: x.score)
        return most_scored.normalized.word

    return list(map(normalize_word, preprocess_text(text).split()))


class TokenDictionary():
    def __init__(self, tokens_decode_table: list):
        self.tokens = tokens_decode_table
        self.rw_tokens = {v: k for k, v in enumerate(self.tokens)}

    def code(self, text):
        for token, token_text in enumerate(self.tokens):
            if text == token_text:
                return token + VOCABULARY_TOKENS_COUNT
        else:
            return VT_UNKOWN

    def decode(self, token):
        if token == VT_UNKOWN:
            return None
        return self.tokens[token - VOCABULARY_TOKENS_COUNT]

    def count(self):
        return len(self.tokens)

    def save(self, path: str):
        with open(path, 'w') as file:
            json.dump(self.tokens, file)

    @staticmethod
    def load(path):
        tokens = None
        with open(path, 'r') as file:
            tokens = json.load(file)
        return TokenDictionary(tokens)
    
    def __str__(self) -> str:
        return "TokenDictionary" + str(self.tokens)


class TokenDictionaryGenerator():
    def __init__(self, vocabulary_size:int):
        self.__vocabulary_size_fixed = vocabulary_size - VOCABULARY_TOKENS_COUNT

    def generate_tokens(self, samples: ds.Dataset) -> TokenDictionary:

        tokens = [',', '.', '-', '?']

        data = []

        morph = MorphAnalyzer()

        for qa in samples.listPairs():
            data += split_text(qa.question, morph)
            data += split_text(qa.answer, morph)

        counter = Counter(data)
        tokens += list(map(lambda x: x[0], counter.most_common(self.__vocabulary_size_fixed - len(tokens))))

        return TokenDictionary(tokens)


class Tokenizer():
    def __init__(self, dictionary: TokenDictionary):
        self.dictionary = dictionary
        self.morph = MorphAnalyzer()

    def tokenize(self, text:str) -> list():
        result = []

        for word in split_text(text, self.morph):
            token = self.dictionary.code(word)

            result.append(token)

        return result
    
    def detokenize(self, tokens:list) -> str:
        result = []

        for token in tokens:
            word = self.dictionary.decode(token)
            if word == None:
                word = "UNKNOWN"
            result.append(word)

        return " ".join(result)

    def decode_token(self, token:int):
        return self.dictionary.decode(token)
    
    def count_tokens(self):
        return self.dictionary.count()

    def get_dictionary(self):
        return self.dictionary


if __name__ == "__main__":
    import Tokens as tk
    import Dataset as ds
    from termcolor import colored

    dataset = ds.load()

    generator = tk.TokenDictionaryGenerator(200)

    tokens = generator.generate_tokens(dataset)

    path = ".aistate/tokens.json"
    tokens.save(path)


    print(colored(f"Tokens were generated and saved into '{path}'", "green"))

    print(colored(f"Tokens: {tokens}", "cyan"))

