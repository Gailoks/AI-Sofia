import torch
from Dataset import *
from AIState import *
from NNRunner import *
from NNTrainer import *
import Tokens as tk
import json

def load_ai_config():
    with open("ai.json") as config:
        config = json.load(config)
    return config

class AI():
    SERVICE_TOKENS = 1
    #ST - service token
    ST_END = 0

    def __init__(self, **options):
        self.options = options

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocabulary_size = options["vocabulary_size"]

        self.ai_state = None
        self.dataset = None

    def load_dataset(self, loader:DatasetLoader, path:str) -> None:
        self.dataset = loader.load(path)

    def create_state(self) -> None:
        generator_options = self.options["tokens_generator"]
        generator_options["vocabulary_size"] = self.vocabulary_size
        generator = tk.TokenDictionaryGenerator(**generator_options)

        tokens = generator.generate_tokens(self.dataset.raw_samples)

        module_options = self.options["module"]
        module = RnnTextModule(device=self.device, input_size=self.vocabulary_size, out_size=self.vocabulary_size + AI.SERVICE_TOKENS, **module_options)

        self.ai_state = AIState(module, tokens)

    def save_state(self, serializer:AIStateSerializer, path:str) -> None:
        serializer.save_state(self.ai_state, path)

    def load_state(self, serializer:AIStateSerializer, path:str) -> None:
        self.ai_state = serializer.load_state(path)

    def create_tokenizer(self) -> tk.Tokenizer:
        return tk.Tokenizer(self.ai_state.tokens)

    def create_runner(self) -> NNRunner:
        return NNRunner(self.device, self.create_tokenizer(), self.ai_state,
            prediction_limit=self.options["runner"]["prediction_limit"],
            end_token=self.__get_service_token(AI.ST_END))
    
    def create_trainer(self) -> NNTrainer:
        iterator = DatasetIterator(self.device, self.dataset, self.create_tokenizer(), end_token=self.__get_service_token(AI.ST_END))
        return NNTrainer(self.ai_state, iterator)
    
    def __get_service_token(self, token_index:int) -> int:
        return self.ai_state.tokens.count() + token_index