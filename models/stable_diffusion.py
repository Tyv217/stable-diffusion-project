import torch
import pytorch_lightning as pl
from diffusers import StableDiffusionPipeline

class StableDiffusionModule(pl.LightningModule):
    def __init__(self, split_gpus = False):
        super().__init__()
        self.split_gpus = split_gpus
        self.model = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            torch_dtype=torch.float16
        )
        self.model.unet = torch.compile(self.model.unet, mode="reduce-overhead", fullgraph=True)
        self.n_steps = 40
        self.high_noise_frac = 0.8

    def forward(self, x):
        image = self.base_model(
            prompt=x,
            num_inference_steps=self.n_steps,
        )[0]
        
        return image
    
    def training_step(self, batch, batch_idx):
        # Extract the input and target from the batch
        x = batch['input']
        y_hat = batch['output']
        y = self.forward(x)
        loss = torch.nn.functional.mse_loss(y, y_hat)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['input']
        y_hat = batch['output']
        y = self.forward(x)
        loss = torch.nn.functional.mse_loss(y, y_hat)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch['input']
        y_hat = batch['output']
        y = self.forward(x)
        loss = torch.nn.functional.mse_loss(y, y_hat)
        self.log('test_loss', loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x = batch['input']
        return self(x)
        
    

