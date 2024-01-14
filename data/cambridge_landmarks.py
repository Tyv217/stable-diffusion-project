import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.transform = transform
        prompts = {
            "GreatCourt": "Great Court, Cambridge", 
            "KingsCollege": "King's College, Cambridge", 
            "OldHospital": "Old Hospital, Cambridge",
            "ShopFacade": "Shop Facade, Cambridge",
            "StMarysChurch": "St Marys Church, Cambridge",
            "Street": "Street view, Cambridge"
        }

        self.imgs = []

        for prompt in prompts.keys():
            img_names = os.listdir(img_dir + "/" + prompt)
            for img_name in img_names:
                self.imgs.append((prompts[prompt], prompt + "/" + img_name)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img[1])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return {"input": img[0], "output": image}

class CambridgeLandmarksData(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  
            transforms.ToTensor(),          
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ])

    def setup(self, stage = None):
        full_dataset = ImageDataset(self.data_dir, transform=self.transform)
        train_size = int(len(full_dataset) * 0.9)
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        pass