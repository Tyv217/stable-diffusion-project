import pytorch_lightning as pl
import torch

from models import StableDiffusionModule

def main():
    model = StableDiffusionModule().to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        image = model.forward("King's College, Cambridge")
    import pdb
    pdb.set_trace()
    trainer = pl.Trainer(gpus=1, max_epochs=1)
    trainer.fit(model)


if __name__ == "__main__":
    main()