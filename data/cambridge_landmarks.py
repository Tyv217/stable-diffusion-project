import pytorch_lightning as pl

class CambridgeLandmarksData(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
    
    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        pass
    
    def train_dataloader(self):
        pass
    
    def val_dataloader(self):
        pass
    
    def test_dataloader(self):
        pass