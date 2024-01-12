import torch
import pytorch_lightning as pl
import pandas as pd

from datasets import Dataset
from models import StableDiffusionModule

def main():
    prompts = [
        "Cambridge University",
        "A Cambridge college",
        "Mathematical Bridge, Cambridge",
        "Street view of Cambridge",
        "Cambridge University Library",
        "A Cambridge student"
    ]
    input_data = []
    for _ in range(10):
        for prompt in prompts:
            input_data.append({"input": prompt})

    dataset = Dataset.from_pandas(pd.DataFrame(input_data))

    model = StableDiffusionModule()
    trainer = pl.Trainer(accelerator = "gpu", gpus = -1)
    print(trainer)
    predictions = trainer.predict(model, dataset)

    prompt_count = {prompt: 0 for prompt in prompts}

    for input, prediction in zip(input_data, predictions):
        prompt_count[input] += 1
        prediction.save(f"{input}_{prompt_count[input]}.png")
    



if __name__ == "__main__":
    main()