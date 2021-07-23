import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


class BaseModel(pl.LightningModule):
    LR = "lr"
    OPTIMIZER = "optimizer"
    MOMENTUM = "momentum"
    WEIGHT_DECAY = "weight_decay"
    DROPOUT = "dropout"

    EARLY_STOPPING = "val/loss"
    EARLY_STOPPING_MODE = "min"
    MODEL_CHECKPOINT = "val/loss"
    MODEL_CHECKPOINT_MODE = "min"

    def __init__(self,
                 batch_size=32,
                 num_workers=4,
                 train_ds=None,
                 val_ds=None,
                 test_ds=None,
                 **model_config):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds

        self.config = model_config

    def configure_optimizers(self):
        optimizer = None
        if self.OPTIMIZER in self.config:
            o = self.config[self.OPTIMIZER]
            if o == "sgd":
                print("Using Optimizer: SGD")
                optimizer = torch.optim.SGD(self.parameters(), lr=self.config[self.LR],
                                            momentum=self.config[self.MOMENTUM],
                                            weight_decay=self.config[self.WEIGHT_DECAY])
            elif o == "adam":
                print("Using Optimizer: Adam")
                optimizer = torch.optim.Adam(self.parameters(), lr=self.config[self.LR],
                                             weight_decay=self.config[self.WEIGHT_DECAY])
        if optimizer is None:
            raise ModuleNotFoundError(f"Optimizer named {o} was not found!")
        return optimizer

    def train_dataloader(self):
        if self.test_ds is None: return None
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        if self.val_ds is None: return None
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def test_dataloader(self):
        if self.test_ds is None: return None
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def get_check_size(self):
        return (2, 1, 22050 * 5)
