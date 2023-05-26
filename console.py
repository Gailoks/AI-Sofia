from AI import *

config = load_ai_config()

ai = AI(**config)

ai.load_nnmodel(".model/model.pkl")
ai.load_tokens(".model/tokens.json")

while True:
    promt = input("Input promt: ")
    print(ai.evaluate(promt))
