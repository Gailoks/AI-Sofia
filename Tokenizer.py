from collections import Counter


class Tokenizer():
    def __init__(self, VOCAB_SIZE) -> None:
        self.vocab_size = VOCAB_SIZE
        self.tokens = {}
        self.rw_tokens = {}

    def fit(self, samples: list(), *special_toks):
        alphabet = set()
        for sample in samples:
            alphabet = alphabet.union(sample)
        for i, char in enumerate(list(special_toks)+list(alphabet)):
            self.tokens[char] = i
        self.tok_dik(samples)

    def tok_dik(self, samples):

        def check(s_dict, key):
            if key in s_dict:
                s_dict[key] += len(key)
            else:
                s_dict[key] = len(key)

        s_dict = {}
        for sample in samples:
            sample = sample.replace("\n", " ")
            sample = "".join(filter(lambda x: x not in ",.?!\r", sample))
            [check(s_dict, word)for word in sample.split()]
            for i in range(1, len(sample)):
                check(s_dict, sample[i-1:i+1])
                check(s_dict, sample[i-2:i+1])
                check(s_dict, sample[i-3:i+1])
                check(s_dict, sample[i-4:i+1])

        length_of_tokens = len(self.tokens)
        sds = sorted(s_dict.items(), key=lambda x: x[1], reverse=True)

        for i in range(self.vocab_size-length_of_tokens-1):
            self.tokens[sds[i][0]] = i+length_of_tokens
        self.rw_tokens = {v: k for k, v in self.tokens.items()}

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


"""with open(r"D:\Projects\RnnTextGen\text.txt",
          encoding="utf-8") as text:
    text = text.read().lower()
tokenizer = Tokenizer(1000)
tokenizer.fit([text])
print(tokenizer.rw_tokens)
"""
