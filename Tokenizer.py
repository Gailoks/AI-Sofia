
class Tokenizer():
    def __init__(self, vocab_size, options):
        self.vocab_size = vocab_size
        self.options = options
        self.rw_tokens = {}
        self.tokens = {}
        
        for token in self.options["special_tokens"]:
            self.add_token(token)


    def fit(self, samples: list()):

        #Get unique letter in samples and add they as tokens
        alphabet = set()
        for sample in samples:
            alphabet = alphabet.union(sample)
        
        for char in list(alphabet):
            self.add_token(char)

        #Generate token using statistic based algoritm
        self.generate_tokens(samples)

    def generate_tokens(self, samples):

        def increase(s_dict, key):
            if key in s_dict:
                s_dict[key] += len(key)
            else:
                s_dict[key] = len(key)

        def increase_limit(s_dict, key, limit):
            if len(key) >= limit:
                increase(s_dict, key)

        s_dict = {}
        for sample in samples:
            sample = sample.replace("\n", " ")
            sample = "".join(filter(lambda x: x not in ",.?!\r", sample))

            if self.options["use_words"]:
                [increase_limit(s_dict, word, self.options["token_depth"]) for word in sample.split()]

            for i in range(1, len(sample)):
                for offset in range(1, self.options["token_depth"]):
                    increase(s_dict, sample[i - offset:i + 1])

        length_of_tokens = len(self.tokens)
        sds = sorted(s_dict.items(), key=lambda x: x[1], reverse=True)

        #Copy first self.vocab_size-length_of_tokens from sds to token to fill token up to vocab_size
        for i in range(self.vocab_size-length_of_tokens):
            self.add_token(sds[i][0])

        self.regenerate_rw_tokens()

    def tokenize(self, text: str) -> list():
        lot = len(self.rw_tokens)
        while text:
            for i in range(lot):
                if text.startswith(self.rw_tokens[lot-i-1]):
                    text = text.removeprefix(self.rw_tokens[lot-i-1])
                    yield lot-i-1
                    break
            else:
                text = text[1:]
                yield 1

    def add_token(self, token):
        self.tokens[token] = len(self.tokens)

    def regenerate_rw_tokens(self):
        self.rw_tokens = {v: k for k, v in self.tokens.items()}


if __name__ == "main":
    with open(r"text.txt", encoding="utf-8") as text:
        text = text.read().lower()

    tokenizer = Tokenizer(1200)
    tokenizer.fit([text])

    print(tokenizer.rw_tokens)