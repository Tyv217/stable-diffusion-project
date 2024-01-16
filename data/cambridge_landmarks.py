import os
import torch
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from diffusers import DiffusionPipeline
from huggingface_hub.utils import insecure_hashlib
from PIL.ImageOps import exif_transpose

class ImageDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        class_num=None,
        size=512,
        center_crop=False,
        tokenizer_max_length=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        text_inputs = self.tokenize_prompt(
            self.tokenizer, self.instance_prompt, tokenizer_max_length=self.tokenizer_max_length
        )
        example["instance_prompt_ids"] = text_inputs.input_ids
        example["instance_attention_mask"] = text_inputs.attention_mask

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)

            class_text_inputs = self.tokenize_prompt(
                self.tokenizer, self.class_prompt, tokenizer_max_length=self.tokenizer_max_length
            )
            example["class_prompt_ids"] = class_text_inputs.input_ids
            example["class_attention_mask"] = class_text_inputs.attention_mask

        return example

    def tokenize_prompt(self, tokenizer, prompt, tokenizer_max_length=None):
        if tokenizer_max_length is not None:
            max_length = tokenizer_max_length
        else:
            max_length = tokenizer.model_max_length

        text_inputs = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        return text_inputs

class CambridgeLandmarksData(pl.LightningDataModule):
    def __init__(self, 
        train_target,
        instance_data_dir,
        tokenizer,
        model_name,
        device,
        class_data_root=None,
        class_num=5,
        size=512,
        center_crop=False,
        tokenizer_max_length=None,
        batch_size=2,
        precision="fp32"
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = device
        self.class_num = class_num
        self.size = size
        self.center_crop = center_crop
        self.tokenizer_max_length = tokenizer_max_length
        self.batch_size = batch_size
        
        prompts = {
            "GreatCourt": "Great Court", 
            "KingsCollege": "King's College", 
            "OldHospital": "Old Hospital",
            "ShopFacade": "Shop Facade",
            "StMarysChurch": "St Marys Church",
            "Street": "Street view"
        }

        prompt = list(prompts.keys())[train_target]

        self.instance_data_root = instance_data_dir + "/" + prompt
        self.class_data_root = class_data_root + "/" + prompt
        self.instance_prompt = prompt + "in Cambridge"
        self.class_prompt = prompt
        self.precision = precision
        train_dataset = ImageDataset(
            instance_data_root=self.instance_data_root,
            instance_prompt=self.instance_prompt,
            class_data_root=self.class_data_root,
            class_prompt=self.class_prompt,
            class_num=self.class_num,
            tokenizer=self.tokenizer,
            center_crop=self.center_crop,
            tokenizer_max_length=self.tokenizer_max_length,
        )
        self.train_dataset = train_dataset
        self.transform = train_dataset.image_transforms
        
        if(len(self.train_dataset.class_images_path) < self.class_num):
            self.generate_class_images(self.class_num - len(self.train_dataset.class_images_path))

        train_dataset = ImageDataset(
            instance_data_root=self.instance_data_root,
            instance_prompt=self.instance_prompt,
            class_data_root=self.class_data_root,
            class_prompt=self.class_prompt,
            class_num=self.class_num,
            tokenizer=self.tokenizer,
            center_crop=self.center_crop,
            tokenizer_max_length=self.tokenizer_max_length,
        )
        self.train_dataset = train_dataset


    def setup(self, stage = None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda examples: self.collate_fn(examples)
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        pass

    def collate_fn(self, examples):
        has_attention_mask = "instance_attention_mask" in examples[0]

        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        if has_attention_mask:
            attention_mask = [example["instance_attention_mask"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

        if has_attention_mask:
            attention_mask += [example["class_attention_mask"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = torch.cat(input_ids, dim=0)

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }

        if has_attention_mask:
            attention_mask = torch.cat(attention_mask, dim=0)
            batch["attention_mask"] = attention_mask

        return batch

    def generate_class_images(self, num_class_images):
        class_images_dir = Path(self.class_data_root)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)

        torch_dtype = torch.float32
        if self.precision == "fp32":
            torch_dtype = torch.float32
        elif self.precision == "fp16":
            torch_dtype = torch.float16
        elif self.precision == "bf16":
            torch_dtype = torch.bfloat16
            
        pipeline = DiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )
        pipeline.set_progress_bar_config(disable=True)

        sample_dataset = PromptDataset(self.class_prompt, num_class_images)
        sample_dataloader = DataLoader(sample_dataset, batch_size=self.batch_size)

        pipeline.to(self.device)

        for example in sample_dataloader:
            image = pipeline(example["prompt"]).images[0]

            hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
            image_filename = class_images_dir / f"{example['index']}-{hash_image}.jpg"
            image.save(image_filename)

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class PromptDataset(Dataset):

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example