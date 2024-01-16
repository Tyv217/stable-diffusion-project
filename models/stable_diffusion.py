import itertools
import torch
import pytorch_lightning as pl
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DiffusionPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

import argparse
import copy
import gc
import importlib
import itertools
import logging
import math
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, model_info, upload_folder
from huggingface_hub.utils import insecure_hashlib
from packaging import version
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

class UInt8Transform():
    """ Convert images to UInt8Tensor. """

    def __call__(self, float_tensor):
        uint8_tensor = (float_tensor * 255).type(torch.uint8)
        return uint8_tensor

class StableDiffusionModule(pl.LightningModule):
    def __init__(self, device, max_training_steps, model_size, precision, output_dir, logging_dir, snr_gamma = 5.0, prior_loss_weight = 1.0, max_grad_norm = 1.0):
        super().__init__()
        self.precision = torch.float32 if precision == 32 else torch.float16
        if model_size != "small":
            self.model_name = "CompVis/stable-diffusion-v1-4"
        else:
            self.model_name = "nota-ai/bk-sdm-small"

        self.uint8transform = UInt8Transform()
        self.n_steps = 40
        self.fid = FrechetInceptionDistance(feature=64, reset_real_features=False)
        self.inception = InceptionScore(feature=64)
        accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision="no",
            log_with="tensorboard",
            project_config=accelerator_project_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            subfolder="tokenizer",
            use_fast=False,
        )
        text_encoder_cls = self.import_model_class_from_model_name_or_path(self.model_name)
        self.text_encoder = text_encoder_cls.from_pretrained(
            self.model_name,
            subfolder="text_encoder",
        )
        self.text_encoder.to(device)
        self.vae = AutoencoderKL.from_pretrained(
            self.model_name, 
            subfolder="vae"
        )
        self.vae.to(device)
        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_name, 
            subfolder="unet"
        )
        self.unet.to(device)
        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )
        self.max_training_steps = max_training_steps
        self.snr_gamma = snr_gamma
        self.prior_loss_weight = prior_loss_weight
        self.max_grad_norm = max_grad_norm
        self.real_images = []
        self.fake_images = []
        torch.backends.cuda.matmul.allow_tf32 = True

    def configure_optimizers(self):
        params_to_optimize = (
            itertools.chain(self.unet.parameters(), self.text_encoder.parameters())
        )
        optimizer = torch.optim.AdamW(
            params_to_optimize,
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
        return [optimizer], [lr_scheduler]

    def forward(self, x):
        
        self.model.text_encoder.to(self.unet.device)
        self.model.unet.to(self.unet.device)
        self.model.vae.to(self.unet.device)
        self.model.safety_checker.to(self.unet.device)
        try:
            image = self.model(prompt=x,num_inference_steps=self.n_steps)[0][0]
        except:
            import pdb
            pdb.set_trace()
    
        
        return image
    
    def training_step(self, batch, batch_idx):
        self.unet.train()
        self.text_encoder.train()
        with self.accelerator.accumulate(self.unet):
            
            pixel_values = batch["pixel_values"].to(dtype=self.precision)
            for pixel_value in pixel_values:
                self.real_images.append(self.uint8transform(pixel_value.detach().clone().cpu()))

            if self.vae is not None:
                # Convert images to latent space
                model_input = self.vae.encode(batch["pixel_values"].to(dtype=self.precision)).latent_dist.sample()
                model_input = model_input * self.vae.config.scaling_factor
            else:
                model_input = pixel_values

            noise = torch.randn_like(model_input)
            bsz, channels, height, width = model_input.shape
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
            )
            timesteps = timesteps.long()

            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_model_input = self.noise_scheduler.add_noise(model_input, noise, timesteps)

            encoder_hidden_states = self.encode_prompt(
                self.text_encoder,
                batch["input_ids"],
                batch["attention_mask"],
                text_encoder_use_attention_mask=True,
            )

            if self.unwrap_model(self.unet).config.in_channels == channels * 2:
                noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)

            # Predict the noise residual
            model_pred = self.unet(noisy_model_input, timesteps, encoder_hidden_states, class_labels=None, return_dict=False)[0]

            if model_pred.shape[1] == 6:
                model_pred, _ = torch.chunk(model_pred, 2, dim=1)

            # Get the target for loss depending on the prediction type
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(model_input, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

            # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)
            # Compute prior loss
            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            
            timesteps, timesteps_prior = torch.chunk(timesteps, 2, dim=0)
            snr = compute_snr(self.noise_scheduler, timesteps)
            base_weight = (
                torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )

            if self.noise_scheduler.config.prediction_type == "v_prediction":
                # Velocity objective needs to be floored to an SNR weight of one.
                mse_loss_weights = base_weight + 1
            else:
                # Epsilon and sample both use the same loss weights.
                mse_loss_weights = base_weight

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")

            # import pdb
            # pdb.set_trace()

            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

            loss = loss + self.prior_loss_weight * prior_loss

            if self.accelerator.sync_gradients:
                params_to_clip = (
                    itertools.chain(self.unet.parameters(), self.text_encoder.parameters())
                )
                self.accelerator.clip_grad_norm_(params_to_clip, self.max_grad_norm)

        self.log('train_loss', loss)
        return loss

    def unwrap_model(self, model):
        model = self.accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def encode_prompt(self, text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
        text_input_ids = input_ids.to(text_encoder.device)

        if text_encoder_use_attention_mask:
            attention_mask = attention_mask.to(text_encoder.device)
        else:
            attention_mask = None

        prompt_embeds = text_encoder(
            text_input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )
        prompt_embeds = prompt_embeds[0]

        return prompt_embeds

    def create_model(self):
        if self.accelerator.is_main_process:
            pipeline_args = {}

            if self.text_encoder is not None:
                pipeline_args["text_encoder"] = self.unwrap_model(self.text_encoder)


            pipeline = DiffusionPipeline.from_pretrained(
                self.model_name,
                unet=self.unwrap_model(self.unet),
                **pipeline_args,
            )

            # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
            scheduler_args = {}

            if "variance_type" in pipeline.scheduler.config:
                variance_type = pipeline.scheduler.config.variance_type

                if variance_type in ["learned", "learned_range"]:
                    variance_type = "fixed_small"

                scheduler_args["variance_type"] = variance_type

            pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)

            self.model = pipeline
            self.model.text_encoder.to(self.unet.device)
            self.model.unet.to(self.unet.device)
            self.model.vae.to(self.unet.device)

    def predict_step(self, batch, batch_idx):
        x = batch['input']
        y = self(x)
        y_tensor = self.uint8transform(self.transform(y)).clone().cpu()
        self.fake_images.append(y_tensor)
        return self(x)

    def get_inception_score(self):
        fake_images = torch.stack(self.fake_images, dim=0)
        self.inception.update(fake_images)
        score = self.inception.compute()
        self.inception.reset()
        return score

    def get_fid_score(self):
        real_images = torch.stack(self.real_images, dim=0)
        fake_images = torch.stack(self.fake_images, dim=0)
        self.fid.update(real_images, real=True)
        self.fid.update(fake_images, real=False)
        score = self.fid.compute()
        self.fid.reset()
        return score

    def reset_images(self):
        self.fake_images = []
        return

    def import_model_class_from_model_name_or_path(self, pretrained_model_name_or_path):
        text_encoder_config = PretrainedConfig.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder"
        )
        model_class = text_encoder_config.architectures[0]

        if model_class == "CLIPTextModel":
            from transformers import CLIPTextModel

            return CLIPTextModel
        elif model_class == "T5EncoderModel":
            from transformers import T5EncoderModel

            return T5EncoderModel
        else:
            raise ValueError(f"{model_class} is not supported.")
        
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