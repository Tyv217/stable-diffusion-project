import torch
import pytorch_lightning as pl
import pandas as pd

from datasets import Dataset
from models import StableDiffusionModule, StableDiffusionLargeModule
from data import CambridgeLandmarksData

import argparse

def main():
    torch.set_float32_matmul_precision('medium')

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_target', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--instance_data_dir', type=str, default="/home/xty20/stable-diffusion-project/data/data_files")
    parser.add_argument('--class_data_dir', type=str, default="/home/xty20/stable-diffusion-project/data/class_data_files")
    parser.add_argument('--output_dir', type=str, default="/home/xty20/stable-diffusion-project/output")
    parser.add_argument('--logging_dir', type=str, default="/home/xty20/stable-diffusion-project/logs")
    parser.add_argument('--max_training_steps', type=int, default=2000)
    parser.add_argument('--model_size', type=str, default="base")
    parser.add_argument('--precision', type=int, default=32)

    args = parser.parse_args()
    
    trainer = pl.Trainer(accelerator = "gpu", max_epochs=50, enable_checkpointing=False)
    
    model = StableDiffusionModule(
        device = "cuda" if torch.cuda.is_available() else "cpu",
        max_training_steps = args.max_training_steps,
        model_size = args.model_size,
        precision = args.precision,
        output_dir = args.output_dir,
        logging_dir = args.logging_dir
    )

    training_data = CambridgeLandmarksData(
        train_target = args.train_target,
        model_name = model.model_name,
        device = "cuda" if torch.cuda.is_available() else "cpu",
        instance_data_dir = args.instance_data_dir,
        class_data_root= args.class_data_dir,
        tokenizer = model.tokenizer
    )

    model.transform = training_data.transform

    prompts = [
        "Great Court in Cambridge",
        "King's College in Cambridge",
        "Old Hospital in Cambridge",
        "Street view in Cambridge",
    ]

    input_data = []
    for _ in range(3):
        for prompt in prompts:
            input_data.append({"input": prompt})

    dataset = Dataset.from_pandas(pd.DataFrame(input_data))

    model.create_model()
    predictions = trainer.predict(model, dataset)

    prompt_count = {prompt: 0 for prompt in prompts}

    for input, prediction in zip(input_data, predictions):
        prompt_count[input["input"]] += 1
        prediction.save(f"visual/before_{input['input']}_{prompt_count[input['input']]}_{args.model_size}.png")

    trainer.fit(model, training_data)

    print(f"Fid Score: {model.get_fid_score()}")
    print(f"Inception Score: {model.get_inception_score()}")

    model.reset_images()
    model.create_model()

    predictions = trainer.predict(model, dataset)

    prompt_count = {prompt: 0 for prompt in prompts}

    for input, prediction in zip(input_data, predictions):
        prompt_count[input["input"]] += 1
        prediction.save(f"visual/after_{input['input']}_{prompt_count[input['input']]}_{args.train_target}_{args.model_size}.png")
    
    print(f"Fid Score: {model.get_fid_score()}")
    print(f"Inception Score: {model.get_inception_score()}")



if __name__ == "__main__":
    main()

# python main.py --train --data_dir="home/xty20/stable-diffusion-project/data/data_files"