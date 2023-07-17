import Seq2Seqmodule as S2S
import json
import Dataset as ds
import Tokens as tk
from termcolor import colored
import torch
import train
import evaluate_model

class CommandModule():
    def __init__(self):
        self.encoder:S2S.Encoder = None
        self.decoder:S2S.Decoder = None

        self.device = "cpu"

        self.dataset:ds.Dataset = None
        self.tokenizer:tk.Tokenizer = None

        with open("ai.json") as config:
            self.ai_config = json.load(config)

    def load_nn(self):
        self.encoder = torch.load(".aistate/encoder.pkl").to(self.device)
        self.decoder = torch.load(".aistate/decoder.pkl").to(self.device)
        return colored("NN models are loaded from '.aistate/'", 'green')

    def create_newnn(self):
        self.encoder = S2S.Encoder(**self.ai_config).to(self.device)
        self.decoder = S2S.Decoder(**self.ai_config).to(self.device)
        return colored("NN models are created", 'green')

    def set_device(self, device:str):
        self.device = device
        return colored(f"New device is {device}", 'green') + '\n' + colored("WARNING: created structs (aka NN models) has not moved to new device, fix it manualy", 'red', attrs=["bold"])

    def load_dataset(self):
        self.dataset = ds.load("../samples")
        return colored("Dataset is loaded", 'green')

    def load_tokens(self):
        tokens = tk.TokenDictionary.load(".aistate/tokens.json")
        self.tokenizer = tk.Tokenizer(tokens)
        return colored("Tokens are loaded from '.aistate/tokens.json'", 'green')

    def generate_tokens(self):
        tokens = tk.TokenDictionaryGenerator(int(self.ai_config["vocabulary_size"]))
        self.tokenizer = tk.Tokenizer(tokens)
        return colored("Tokens are generated from data, you should save it manualy", 'green')

    def save_tokens(self):
        path = ".aistate/tokens.json"
        self.tokens.save(path)
        return colored(f"Tokens are saved to '{path}'", 'green')
    
    def save_nn(self):
        torch.save(self.encoder, ".aistate/encoder.pkl")
        torch.save(self.decoder, ".aistate/decoder.pkl")
    
    def train(self, epoch_count:int, batch_size:int, out_len:int):
        train.train(epoch_count, self.encoder, self.decoder, self.tokenizer, self.dataset, self.device, batch_size, out_len)

    def eval(self, promt:str):
        promt = promt.replace('_', ' ')
        ans, _ = evaluate_model.evaluate(self.encoder, self.decoder, self.tokenizer, promt, max_length=120, device=self.device)
        return ans
