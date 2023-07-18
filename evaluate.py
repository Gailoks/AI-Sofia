"""Module to evaluate AI"""
import torch
import neuralnetwork
import tokens
import servicetokens as st


def evaluate(
        network: neuralnetwork.NeuralNetwork,
        tokenizer: tokens.Tokenizer,
        text: str,
        device: str = 'cpu'
    ):
    """Evaluates ai with promt"""

    network.to(device)

    service = st.ServiceTokens(tokenizer.count_tokens())

    sto_switch = torch.LongTensor([service.get(st.STO_SWITCH)]).to(device)
    null_input = torch.zeros(network.hidden_size).to(device)

    predicted_tokens = []

    text = tokenizer.tokenize(text)
    text = torch.LongTensor(text).to(device)

    encoder_outs, _ = network.encoder(text)

    decoder_hidden = None

    READ_MODE = True
    WRTIE_MODE = False

    current_mode = READ_MODE
    pointer = -1

    while True:
        if current_mode == READ_MODE:
            pointer += 1
            if pointer >= encoder_outs.size()[0]:
                break

            encoder_out = encoder_outs[pointer]
            decoder_out, decoder_hidden = network.decoder(encoder_out, decoder_hidden)

            argmax = decoder_out.argmax()
            if argmax == sto_switch:
                current_mode = WRTIE_MODE

        if current_mode == WRTIE_MODE:
            decoder_out, decoder_hidden = network.decoder(null_input, decoder_hidden)

            argmax = decoder_out.argmax()
            if argmax == sto_switch:
                current_mode = READ_MODE
            else:
                predicted_tokens.append(int(argmax))

    return tokenizer.detokenize(predicted_tokens)
