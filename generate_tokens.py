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