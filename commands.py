"""Commands module for AI-SOFIA"""
import torch
from termcolor import colored
import config
import neuralnetwork
import texts
import train
import tokens
from evaluate import evaluate

class CommandModule():
    """Commands module for AI-SOFIA"""
    def __init__(self):
        self.configuration = config.config()

        self.neuralnetwork: neuralnetwork.NeuralNetwork = None
        self.device = "cpu"
        self.texts: texts.TextsDatabase = None
        self.dataset: texts.Dataset = None
        self.tokenizer: tokens.Tokenizer = None

    def load_nn(self):
        """Loads neural network from files"""
        self.neuralnetwork = torch.load(".state/nn.pkl")
        return colored("NN models are loaded from '.state/'", 'green')

    def create_newnn(self):
        self.neuralnetwork = neuralnetwork.NeuralNetwork(self.configuration.scope("neural model"))
        return colored("NN models are created", 'green')

    def set_device(self, device:str):
        self.device = device
        return colored(f"New device is {device}", 'green')

    def load_texts(self):
        self.texts = texts.load_text_database("samples")
        return colored("Text database is loaded from 'samples/'", 'green')

    def parse_dataset(self):
        self.dataset = texts.load_dataset(self.texts)
        return colored("Dataset is parsed from text database", 'green')

    def load_tokens(self):
        tokens_dic = tokens.TokenDictionary.load(".state/tokens.json")
        self.tokenizer = tokens.Tokenizer(tokens_dic)
        return colored("Tokens are loaded from '.state/tokens.json'", 'green')

    def generate_tokens(self):
        tokens_gen = self.configuration.resolve(tokens.TokenDictionaryGenerator, "token generator")
        tokens_dic = tokens_gen.generate_tokens(self.texts)
        self.tokenizer = tokens.Tokenizer(tokens_dic)
        return colored("Tokens are generated from data, you should save it manualy", 'green')

    def save_tokens(self):
        path = ".state/tokens.json"
        self.tokenizer.get_dictionary().save(path)
        return colored(f"Tokens are saved to '{path}'", 'green')

    def save_nn(self):
        torch.save(self.neuralnetwork, ".state/nn.pkl")

    def train(self, epoch_count:int):
        trainer = self.configuration.resolve(train.Trainer, "trainer")
        trainer.train(self.neuralnetwork, self.tokenizer, self.dataset, epoch_count, self.device)

    def eval(self, promt:str):
        promt = promt.replace('_', ' ')
        ans = evaluate(self.neuralnetwork, self.tokenizer, promt, device=self.device)
        return ans
