import json
import Tokens as tk
import torch
import torch.nn as nn
import numpy as np
import random
from io import StringIO


class StringBuilder:
    string = None

    def __init__(self):
        self.string = StringIO()

    def append(self, str: str):
        self.string.write(str)

    def build(self):
        return self.string.getvalue()


def load_ai_config():
    with open("ai.json") as config:
        config = json.load(config)
    return config


class AI:
    SERVICE_TOKENS = 1
    # ST - Service token
    ST_END = 0

    """
    Accepts options:\n
    vocabulary_size (required) - size of token dictionary\n
    generator.token_depth (default: 5) - token generator's max lenght of token in statistic analize\n
    generator.use_words (default: True) - token generator's also analize full words
    prediction_limit (default: 20) - AI output tokens limit
    """

    def __init__(self, **options):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.options = options
        self.vocabulary_size = options["vocabulary_size"]

        self.tokenizer: tk.Tokenizer = None
        self.dataset: list(AI.__QAPair) = None
        self.samples = []

        self.model: RnnTextGen = None

    def add_sample(self, sample: str):
        self.samples.append(sample)

    def add_samples(self, samples):
        self.samples += samples

    def generate_dataset(self):
        self.dataset = []
        for sample in self.samples:
            lines = sample.splitlines()
            questions = lines[::2]
            answers = lines[1::2]
            for q, a in zip(questions, answers):
                self.dataset.append(AI.__QAPair(q, a))

    def generate_tokens(self):
        tokens_generator = tk.TokenDictionaryGenerator(
            vocabulary_size=self.vocabulary_size,
            token_depth=self.options.get("generator.token_depth", 5),
            use_words=self.options.get("generator.use_words", True))

        tokens = tokens_generator.generate_tokens(self.samples)
        self.tokenizer = tk.Tokenizer(tokens)

    def load_tokens(self, tokens_path: str):
        tokens = tk.TokenDictionary.load(tokens_path)
        self.tokenizer = tk.Tokenizer(tokens)

    def save_tokens(self, tokens_path: str):
        self.tokenizer.get_dictionary().save(tokens_path)

    def create_new_nnmodel(self):
        self.model = RnnTextGen(self.vocabulary_size + AI.SERVICE_TOKENS, 1000,
                                500, 2, self.vocabulary_size + AI.SERVICE_TOKENS).to(self.device)
        self.model.bind(self)

    def load_nnmodel(self, pkl_file: str):
        self.model = torch.load(pkl_file).to(self.device)
        self.model.bind(self)

    def save_nnmodel(self, pkl_file: str):
        self.model.prepare_serialize()
        torch.save(self.model, pkl_file)
        self.model.bind(self)

    def evaluate(self, text: str):
        return self.model.evaluate(text, self.options.get("prediction_limit", 20))

    """
    batch_size - n/a, ASK Gailoks, I don't know
    """

    def train(self, batch_size: int):
        return self.model.train(batch_size)

    def get_service_token(self, service_index: int):
        return self.vocabulary_size + service_index

    class EpochResult:
        def __init__(self, epoch, mean_loss):
            self.epoch = epoch
            self.mean_loss = mean_loss

    class __QAPair:
        def __init__(self, question, answer):
            self.question = question
            self.answer = answer

# WARNING: rename action will broke all .pkl files


class RnnTextGen(nn.Module):
    def __init__(self, input_size, inp_lstm_size, hid_size, n_layers, out_size, dropout=0.2):
        super().__init__()
        self.input_size = input_size
        self.n_layers = n_layers
        self.hidden_size = hid_size
        self.Encoder = nn.Embedding(input_size, inp_lstm_size)
        self.lstm = nn.LSTM(inp_lstm_size, hid_size, n_layers)
        self.dropout = nn.Dropout(dropout)
        self.l1 = nn.Linear(hid_size, out_size)

        self.__ai: AI = None
        self.__end_tensor = None

    def bind(self, ai: AI):
        self.__ai = ai
        self.__end_tensor = torch.LongTensor(
            [self.__ai.get_service_token(AI.ST_END)]).to(self.__ai.device)

    def prepare_serialize(self):
        self.__ai = None
        self.__end_tensor = None

    def forward(self, x, hidden=None):
        x = self.Encoder(x)
        x, hidden = self.lstm(x)
        x = self.dropout(x)
        x = self.l1(x)
        return x, hidden

    def evaluate(self, text: str, prediction_lim: int) -> str:
        text_tensor = torch.LongTensor(
            list(self.__ai.tokenizer.tokenize(text))).to(self.__ai.device)
        hidden = self.__init_hidden()
        predicted_text = StringBuilder()

        for _ in range(prediction_lim):
            next_w, hidden = self(
                text_tensor.view(-1, 1).to(self.__ai.device), hidden)
            token_tensor = next_w[-1].argmax()

            text_tensor = torch.cat([text_tensor, token_tensor.view(-1)])

            if token_tensor == self.__end_tensor:
                break

            token_text = self.__ai.tokenizer.decode_token(int(token_tensor))
            predicted_text.append(token_text)

        return predicted_text.build()

    def train(self, batch_size: int):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=5,
            verbose=True,
            factor=0.5
        )

        loss_avg = []
        epoch = 0

        while True:
            epoch += 1

            for target, train in self.__get_batch():
                super().train()

                hidden = self.__init_hidden(batch_size)

                output, hidden = self(train, hidden)
                target_len = len(target)
                loss = criterion(output[-target_len:], target)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss_avg.append(loss.item())

                if len(loss_avg) >= 50:
                    mean_loss = np.mean(loss_avg)
                    scheduler.step(mean_loss)
                    self.eval()
                    loss_avg = []

                    yield AI.EpochResult(epoch, mean_loss)

    def __init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(self.__ai.device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(self.__ai.device))

    def __get_batch(self):
        for qa in self.__ai.dataset:
            question_idx = list(self.__ai.tokenizer.tokenize(qa.question))

            target = list(self.__ai.tokenizer.tokenize(qa.answer))
            target.append(self.__ai.get_service_token(AI.ST_END))

            test = question_idx + target[:-1]

            target = torch.LongTensor(target).to(self.__ai.device)
            test = torch.LongTensor(test).to(self.__ai.device)
            yield target, test
