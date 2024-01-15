import torch
import pytorch_lightning as pl
import pandas as pd

from datasets import Dataset
from models import StableDiffusionModule, StableDiffusionLargeModule
from data import CambridgeLandmarksData

import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_target', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--data_dir', type=str, default="data/data_files")

    args = parser.parse_args()
    
    trainer = pl.Trainer(accelerator = "gpu")

    training_data = CambridgeLandmarksData(args.batch_size, args.data_dir)

    prompts = [
        "A university in Cambridge",
        "A college in Cambridge",
        "Mathematical Bridge in Cambridge",
        "Street view in Cambridge",
        "Library in Cambridge",
        "A student in Cambridge"
    ]

    input_data = []
    for _ in range(10):
        for prompt in prompts:
            input_data.append({"input": prompt})

    dataset = Dataset.from_pandas(pd.DataFrame(input_data))

    model = StableDiffusionModule("cuda" if torch.cuda.is_available() else "cpu")
                                  
    predictions = trainer.predict(model, dataset)

    prompt_count = {prompt: 0 for prompt in prompts}

    for input, prediction in zip(input_data, predictions):
        prompt_count[input["input"]] += 1
        prediction.save(f"visual/before_{input['input']}_{prompt_count[input['input']]}.png")

    trainer.train(model, training_data)
        
    predictions = trainer.predict(model, dataset)

    prompt_count = {prompt: 0 for prompt in prompts}

    for input, prediction in zip(input_data, predictions):
        prompt_count[input["input"]] += 1
        prediction.save(f"visual/after_{input['input']}_{prompt_count[input['input']]}.png")
    



if __name__ == "__main__":
    main()

# python main.py --train --data_dir="home/xty20/stable-diffusion-project/data/data_files"