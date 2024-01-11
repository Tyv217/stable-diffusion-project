import pytorch_lightning as pl

from models import StableDiffusionModule


def main():
    model = StableDiffusionModule()
    image = model.predict("King's College, Cambridge")
    import pdb
    pdb.set_trace()
    trainer = pl.Trainer(gpus=1, max_epochs=1)
    trainer.fit(model)


if __name__ == "__main__":
    main()