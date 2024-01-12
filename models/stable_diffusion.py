import torch
import pytorch_lightning as pl
from diffusers import DiffusionPipeline

class StableDiffusionModule(pl.LightningModule):
    def __init__(self, split_gpus = True):
        super().__init__()
        self.split_gpus = split_gpus
        self.base_model = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=torch.float16
        )
        if split_gpus:
            self.base_model.cuda(0)
        self.base_model.unet = torch.compile(self.base_model.unet, mode="reduce-overhead", fullgraph=True)

        self.refiner_model = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base_model.text_encoder_2,
            vae=self.base_model.vae,
            torch_dtype=torch.float16
        )
        if split_gpus:
            self.base_model.cuda(1)
        self.n_steps = 40
        self.high_noise_frac = 0.8

    def forward(self, x):
        print(self.base_model.device)
        print(self.refiner_model.device)
        import pdb
        pdb.set_trace()
        image = self.base_model(
            prompt=x,
            num_inference_steps=self.n_steps,
            denoising_end=self.high_noise_frac,
            output_type="latent",
        ).images

        if self.split_gpus:
            image = image.cuda(1)

        image = self.refiner_model(
            prompt=x,
            num_inference_steps=self.n_steps,
            denoising_start=self.high_noise_frac,
            image=image,
        ).images[0]
        
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
        
    

