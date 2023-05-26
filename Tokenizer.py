import json
# TokenDictionary, TokenDictionaryGenerator and Tokenizer


class TokenDictionary():
    def __init__(self, tokens_decode_table: list):
        self.tokens = sorted(tokens_decode_table,
                             key=lambda x: len(x),
                             reverse=True)
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

    def __init__(self, **options):
        if "vocabulary_size" not in options:
            raise Exception(
                "Required paramter 'vocabulary_size' is not present in options")
        self.options = options

    def generate_tokens(self, samples: list) -> TokenDictionary:
        tokens = []

        # Get unique letter in samples and add they as tokens
        alphabet = set()
        for sample in samples:
            alphabet = alphabet.union(sample)

        for char in alphabet:
            tokens.append(char)

        # Generate token using statistic based algoritm
        def increase(s_dict, key):
            s_dict[key] = s_dict.get(key, 0) + len(key)

        def increase_limit(s_dict, key, limit):
            if len(key) >= limit:
                increase(s_dict, key)

        token_depth = self.options.get("token_depth", 5)
        s_dict = {}
        for sample in samples:
            sample = sample.replace("\n", " ")
            sample = "".join(filter(lambda x: x not in ",.?!\r", sample))

            if self.options.get("use_words", True):
                [increase_limit(s_dict, word, token_depth)
                 for word in sample.split()]

            for i in range(1, len(sample)):
                for offset in range(1, token_depth):
                    increase(s_dict, sample[i - offset:i + 1])

        length_of_tokens = len(tokens)
        sds = sorted(s_dict.items(), key=lambda x: x[1], reverse=True)

        # Copy first self.vocab_size-length_of_tokens from sds to token to fill token up to vocab_size
        for i in range(self.options["vocabulary_size"]-length_of_tokens):
            tokens.append(sds[i][0])

        return TokenDictionary(tokens)


class Tokenizer():
    def __init__(self, dictionary: TokenDictionary, **options):
        self.dictionary = dictionary
        self.options = options

    def tokenize(self, text: str) -> list():
        while text:
            token, char_ate = self.dictionary.code(text)
            if token == -1:
                text = text[1:]
                continue
            text = text[char_ate:]
            yield token
