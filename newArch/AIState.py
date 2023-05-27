import torch
from RnnTextModule import *
import Tokens as tk

class AIState():
    def __init__(self, module:RnnTextModule, tokens:tk.TokenDictionary):
        self.module = module
        self.tokens = tokens

class AIStateSerializer():
    def __init__(self, **options):
        self.options = options

    def save_state(self, state:AIState, path:str):
        torch.save(state.module, path + "/module.pkl")
        state.tokens.save(path + "/tokens.json")

    def load_state(self, path:str) -> AIState:
        module = torch.load(path + "/module.pkl")
        tokens = tk.TokenDictionary.load(path +"/tokens.json")
        return AIState(module, tokens)