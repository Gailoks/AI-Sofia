from AI import *

options = load_ai_config()

ai = AI(**options)

dataset_loader = DatasetLoader()
ai.load_dataset(dataset_loader, "../samples")

ai.create_state()
trainer = ai.create_trainer()

for epoch in trainer.train():
    print(f"Epoch: {epoch.epoch}, Mean loss: {epoch.mean_loss}")
    if epoch.mean_loss <= 0.1:
        break

serializer = AIStateSerializer()
ai.save_state(serializer, ".model")