import torch
import pytorch_lightning as pl
import pandas as pd

from datasets import Dataset
from models import StableDiffusionModule

def main():
    prompt = "King's College, Cambridge"
    input_data = [{"input": prompt} for _ in range(10)]
    dataset = Dataset.from_pandas(pd.DataFrame(input_data))

    model = StableDiffusionModule()
    trainer = pl.Trainer(accelerator = "gpu", gpus = -1)
    print(trainer)
    predictions = trainer.predict(model, dataset)
    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    main()