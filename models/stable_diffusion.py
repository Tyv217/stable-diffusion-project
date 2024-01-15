import torch
import pytorch_lightning as pl
from diffusers import StableDiffusionPipeline, DiffusionPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

class StableDiffusionModule(pl.LightningModule):
    def __init__(self, device, max_training_steps, model_size, precision):
        super().__init__()
        dtype = torch.float32 if precision == 32 else torch.float16
        if model_size == "small":
            self.model = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4", 
                torch_dtype=dtype
        else：
            self.model = StableDiffusionPipeline.from_pretrained(
                "nota-ai/bk-sdm-small", 
                torch_dtype=dtype
            )

        self.model.to(device)
        self.model.unet = torch.compile(self.model.unet, mode="reduce-overhead", fullgraph=True)
        self.n_steps = 40
        self.fid = FrechetInceptionDistance(feature=64, reset_real_features=False)
        self.inception = InceptionScore(feature=64)
        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )
        self.max_training_steps = max_training_steps

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=5e-6,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-08,
        )
        lr_scheduler = get_scheduler(
            'constant',
            optimizer = optimizer,
            num_warmup_steps = 500,
            num_training_steps = self.max_training_steps,
        )
        return optimizer, lr_scheduler    

    def forward(self, x):
        image = self.model(
            prompt=x,
            num_inference_steps=self.n_steps,
        )[0][0]
        
        return image
    
    def training_step(self, batch, batch_idx):
        # Extract the input and target from the batch
        x = batch['input']
        try:
            attention_mask = batch['attention_mask']
        except:
            attention_mask = None
        y_hat = batch['output']
        self.fid.update(y_hat, real=True)
        latents = self.model.vae.encode(x).to(dtype=weight_dtype).latent_dist.sample()
        latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # Sample a random timestep for each image 
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
        )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        text_input_ids = input_ids.to(text_encoder.device)

        prompt_embeds = self.model.text_encoder(
            text_input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )
        encoder_hidden_states = prompt_embeds[0]  

        # Predict the noise residual
        model_pred = self.model.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        
        loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['input']
        y_hat = batch['output']
        y = self.forward(x)
        self.fid.update(y_hat, real=True)
        latents = self.model.vae.encode(x).to(dtype=weight_dtype).latent_dist.sample()
        latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # Sample a random timestep for each image 
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
        )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning  
        encoder_hidden_states = self.model.text_encoder(x)[0]

        # Predict the noise residual
        model_pred = self.model.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        
        loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
        
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
        y = self(x)
        self.fid.update(y, real=False)
        self.inception.update(y)
        return self(x)

    def get_inception_score(self):
        score = self.inception.compute()
        self.inception.reset()
        return score

    def get_fid_score(self):
        score = self.fid.compute()
        self.fid.reset()
        return score
        
class StableDiffusionLargeModule(pl.LightningModule):
    def __init__(self, device):
        super().__init__()
        self.base_model = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=torch.float16,
            use_safetensors=True
        )

        self.base_model.to(device)

        self.base_model.unet = torch.compile(self.base_model.unet, mode="reduce-overhead", fullgraph=True)

        self.refiner_model = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base_model.text_encoder_2,
            vae=self.base_model.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

        self.refiner_model.to(device)

        self.n_steps = 40
        self.high_noise_frac = 0.8

    def forward(self, x):
        unrefined_image = self.base_model(
            prompt=x,
            num_inference_steps=self.n_steps,
            denoising_end=self.high_noise_frac,
            output_type="latent",
        ).images

        image = self.refiner_model(
            prompt=x,
            num_inference_steps=self.n_steps,
            denoising_start=self.high_noise_frac,
            image=unrefined_image,
        ).images[0]
        
        return (image, unrefined_image)
    
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

    def predict_step(self, batch, batch_idx):
        x = batch['input']
        return self(x)