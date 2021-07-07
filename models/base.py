import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

import torch.nn.functional as F
import torchmetrics as tm

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

        print(self.config)

    def configure_optimizers(self):
        optimizer = None
        if self.OPTIMIZER in self.config:
            o = self.config[self.OPTIMIZER]
            if o == "sgd":
                optimizer = torch.optim.SGD(self.parameters(), lr=self.config[self.LR], momentum=self.config[self.MOMENTUM], weight_decay=self.config[self.WEIGHT_DECAY])
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config[self.LR], weight_decay=self.config[self.WEIGHT_DECAY])
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

class BaseCatModel(BaseModel):

    def __init__(self,
                batch_size=32,
                num_workers=4,
                train_ds=None,
                val_ds=None,
                test_ds=None,
                **model_config):
        super().__init__(batch_size, num_workers, train_ds, val_ds, test_ds, **model_config)

        ## metrics
        self.train_acc = tm.Accuracy(top_k=3)

        self.val_acc = tm.Accuracy(top_k=3)
        self.val_f1_class = tm.F1(num_classes=4, average='none')
        self.val_f1_global = tm.F1(num_classes=4)

        self.test_acc = tm.Accuracy(top_k=3)
        self.test_f1_class = tm.F1(num_classes=4, average='none')
        self.test_f1_global = tm.F1(num_classes=4)

        ## loss
        self.loss = F.cross_entropy
    
    def predict(self, x):
        x = self.forward(x)
        return F.softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_logit = self(x)
        loss = self.loss(y_logit, y)
        pred = F.softmax(y_logit, dim=1)

        self.log('train/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train/acc', self.train_acc(pred, y), prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_logit = self(x)
        loss = self.loss(y_logit, y)
        pred = F.softmax(y_logit, dim=1)
        
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc(pred, y), prog_bar=True)

        self.log("val/f1_global", self.val_f1_global(pred, y), on_step=False, on_epoch=True)

        f1_scores = self.val_f1_class(pred, y)
        for (i, x) in enumerate(torch.flatten(f1_scores)):
            self.log("val/f1_class_{}".format(i), x, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_logit = self(x)
        loss = self.loss(y_logit, y)
        pred = F.softmax(y_logit, dim=1)

        self.log("test/loss", loss)
        self.log("test/acc", self.test_acc(pred, y))
        self.log("test/f1_global", self.test_f1_global(pred, y))

        f1_scores = self.test_f1_class(pred, y)
        for (i, x) in enumerate(torch.flatten(f1_scores)):
            self.log("test/f1_class_{}".format(i), x)