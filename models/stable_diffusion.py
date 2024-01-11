import torch
import pytorch_lightning as pl
from diffusers import DiffusionPipeline

class StableDiffusionModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.base_model = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to(self.device)

        self.refiner_model = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base_model.text_encoder_2,
            vae=self.base_model.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to(self.device)

        self.n_steps = 40
        self.high_noise_frac = 0.8

    def forward(self, x):
        image = self.base_model(
            prompt=x,
            num_inference_steps=self.n_steps,
            denoising_end=self.high_noise_frac,
            output_type="latent",
        ).images

        image = self.refiner_model(
            prompt=x,
            num_inference_steps=self.n_steps,
            denoising_start=self.high_noise_frac,
            image=image,
        ).images[0]
        
        return image
    
    def training_step(self, batch, batch_idx):
        # Extract the input and target from the batch
        x, y_hat = batch
        y = self.forward(x)
        loss = torch.nn.functional.mse_loss(y, y_hat)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y_hat = batch
        y = self.forward(x)
        loss = torch.nn.functional.mse_loss(y, y_hat)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y_hat = batch
        y = self.forward(x)
        loss = torch.nn.functional.mse_loss(y, y_hat)
        self.log('test_loss', loss)
        return loss
        
    

