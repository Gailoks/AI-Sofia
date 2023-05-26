from AI import *
import os

config = load_ai_config()

ai = AI(**config)

for sample in os.listdir('samples'):
    with open("samples/" + sample, encoding="utf-8") as text:
        ai.add_sample(text.read().lower())

ai.generate_dataset()
ai.generate_tokens()

ai.create_new_nnmodel()

for epoch in ai.train(batch_size=1):
    print(f"Epoch: {epoch.epoch}, Mean loss: {epoch.mean_loss}")
    if (epoch.epoch >= 20):
        break
 
ai.save_nnmodel(".model/model.pkl")
ai.save_tokens(".model/tokens.json")