import torch
import Tokens as tk
from AIState import *
from StringBuilder import *

class NNRunner():
    def __init__(self, device, tokenizer:tk.Tokenizer, state:AIState, **options):
        self.tokenizer = tokenizer
        self.device = device
        self.options = options
        self.state = state
        self.prediction_limit:int = options["prediction_limit"]
        self.end_token:int = torch.LongTensor(options["end_token"]).to(self.__ai.device)


    def evaluate(self, promt:str) -> str:
        text_tensor = torch.LongTensor(list(self.tokenizer.tokenize(promt))).to(self.device)
        hidden = self.state.module.create_empty_hidden()
        predicted_text = StringBuilder()

        for _ in range(self.prediction_limit):
            next_w, hidden = self(text_tensor.view(-1, 1).to(self.device), hidden)
            token_tensor = next_w[-1].argmax()

            text_tensor = torch.cat([text_tensor, token_tensor.view(-1)])

            if token_tensor == self.end_token:
                break

            token_text = self.__ai.tokenizer.decode_token(int(token_tensor))
            predicted_text.append(token_text)

        return predicted_text.build()