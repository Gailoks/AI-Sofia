import json
import Dataset as ds
# TokenDictionary, TokenDictionaryGenerator and Tokenizer


class TokenDictionary():
    def __init__(self, tokens_decode_table: list):
        self.tokens = sorted(tokens_decode_table, key = lambda x: len(x), reverse = True)
        self.rw_tokens = {v: k for k, v in enumerate(self.tokens)}

    def code(self, text):
        for token, token_text in enumerate(self.tokens):
            if text.startswith(token_text):
                char_ate = len(token_text)
                return token, char_ate
        else:
            return -1, 0

    def decode(self, token):
        return self.tokens[token]

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


class TokenDictionaryGenerator():
    """
    Accept options:\n
    vocabulary_size (required) - target tokens dictionary size\n
    token_depth (default: 5) - max lenght of token in statistic analize\n
    use_words (default: True) - also analize full words
    """

    def __init__(self, vocabulary_size:int, token_depth:int = 5, use_words:bool = True):
        self.__vocabulary_size = vocabulary_size
        self.__token_depth = token_depth
        self.__use_words = use_words

    def generate_tokens(self, samples: ds.Dataset) -> TokenDictionary:
        tokens = []

        # Get unique letter in samples and add they as tokens
        alphabet = set()
        for qa in samples.listPairs():
            alphabet = alphabet.union(qa.question)
            alphabet = alphabet.union(qa.answer)

        for char in alphabet:
            tokens.append(char)

        # Generate token using statistic based algoritm
        def increase(s_dict, key):
            s_dict[key] = s_dict.get(key, 0) + len(key)

        def increase_limit(s_dict, key, limit):
            if len(key) >= limit:
                increase(s_dict, key)

        token_depth = self.__token_depth
        s_dict = {}

        sample = " ".join(map(lambda qa: qa.question + " " + qa.answer, samples.listPairs()))
        sample = "".join(filter(lambda x: x not in ",.?!", sample))

        if self.__use_words:
            [increase_limit(s_dict, word, token_depth)
            for word in sample.split()]

        for i in range(1, len(sample)):
            for offset in range(1, token_depth):
                increase(s_dict, sample[i - offset:i + 1])

        length_of_tokens = len(tokens)
        sds = sorted(s_dict.items(), key=lambda x: x[1], reverse=True)

        # Copy first self.vocab_size - length_of_tokens from sds to token to fill token up to vocab_size
        for i in range(self.__vocabulary_size - length_of_tokens):
            tokens.append(sds[i][0])

        return TokenDictionary(tokens)


class Tokenizer():
    def __init__(self, dictionary: TokenDictionary, **options):
        self.dictionary = dictionary
        self.options = options

    def tokenize(self, text:str) -> list():
        while text:
            token, char_ate = self.dictionary.code(text)
            if token == -1:
                text = text[1:]
                continue
            text = text[char_ate:]
            yield token

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

    generator = tk.TokenDictionaryGenerator(
        vocabulary_size = 1200,
        token_depth = 5,
        use_words = True
    )

    tokens = generator.generate_tokens(dataset)

    path = ".aistate/tokens.json"
    tokens.save(path)

    print(colored(f"Tokens were generated and saved into '{path}'", "green"))