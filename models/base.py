import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from utils.activation import CustomELU

from utils.loss import rmse_loss
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
                optimizer = torch.optim.SGD(self.parameters(), lr=self.config[self.LR], momentum=self.config[self.MOMENTUM], weight_decay=self.config[self.WEIGHT_DECAY])
            elif o == "adam":
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

class BaseStatModel(BaseModel):

    STD_ACTIVATION = "std_activation"

    def __init__(self,
                batch_size=32,
                num_workers=4,
                train_ds=None,
                val_ds=None,
                test_ds=None,
                **model_config):
        super().__init__(batch_size, num_workers, train_ds, val_ds, test_ds, **model_config)

        ## loss
        self.loss = rmse_loss
        
        self.test_arousal_mean_r2 = tm.R2Score(num_outputs=1)
        self.test_valence_mean_r2 = tm.R2Score(num_outputs=1)
        self.test_arousal_std_r2 = tm.R2Score(num_outputs=1)
        self.test_valence_std_r2 = tm.R2Score(num_outputs=1)

        self.test_mean_r2score = tm.R2Score(num_outputs=2)

    def _get_std_activation(self):
        stdActivation = None
        if self.config[self.STD_ACTIVATION] == "custom":
            print("Model: StdActivation uses CustomELU")
            stdActivation = CustomELU(alpha=1.0)
        elif self.config[self.STD_ACTIVATION] == "relu":
            print("Model: StdActivation uses ReLU")
            stdActivation = nn.ReLU()
        elif self.config[self.STD_ACTIVATION] == "softplus":
            print("Model: StdActivation uses Softplus")
            stdActivation = nn.Softplus()
        if stdActivation is None:
            raise Exception("Activation Type Unknown!")
        return stdActivation
    
    def predict(self, x):
        return self.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        pred = self(x)
        loss = self.loss(pred, y)

        self.log('train/loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        pred = self(x)
        loss = self.loss(pred, y)

        arousal_std_rmse = self.loss(pred[:, 3], y[:, 3])
        valence_std_rmse = self.loss(pred[:, 2], y[:, 2])

        arousal_mean_rmse = self.loss(pred[:, 1], y[:, 1])
        valence_mean_rmse = self.loss(pred[:, 0], y[:, 0])

        self.log("val/loss", loss, prog_bar=True)

        self.log('val/arousal_std_rmse', arousal_std_rmse, on_step=False, on_epoch=True)
        self.log('val/valence_std_rmse', valence_std_rmse, on_step=False, on_epoch=True)

        self.log("val/arousal_mean_rmse", arousal_mean_rmse, on_step=False, on_epoch=True)
        self.log("val/valence_mean_rmse", valence_mean_rmse, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch

        pred = self(x)
        loss = self.loss(pred, y)

        arousal_std_rmse = self.loss(pred[:, 3], y[:, 3])
        valence_std_rmse = self.loss(pred[:, 2], y[:, 2])

        arousal_mean_rmse = self.loss(pred[:, 1], y[:, 1])
        valence_mean_rmse = self.loss(pred[:, 0], y[:, 0])

        mean_r2score = self.test_mean_r2score(pred[:, [0, 1]], y[:, [0, 1]])

        arousal_mean_r2score = self.test_arousal_mean_r2(pred[:, 1], y[:, 1])
        valence_mean_r2score = self.test_valence_mean_r2(pred[:, 0], y[:, 0])

        arousal_std_r2score = self.test_arousal_std_r2(pred[:, 3], y[:, 3])
        valence_std_r2score = self.test_valence_std_r2(pred[:, 2], y[:, 2])

        self.log("test/loss", loss)

        self.log('test/mean_r2score', mean_r2score, on_step=False, on_epoch=True)

        self.log('test/arousal_mean_r2score', arousal_mean_r2score, on_step=False, on_epoch=True)
        self.log('test/valence_mean_r2score', valence_mean_r2score, on_step=False, on_epoch=True)

        self.log('test/arousal_std_r2score', arousal_std_r2score, on_step=False, on_epoch=True)
        self.log('test/valence_std_r2score', valence_std_r2score, on_step=False, on_epoch=True)

        self.log("test/arousal_mean_rmse", arousal_mean_rmse, on_step=False, on_epoch=True)
        self.log("test/valence_mean_rmse", valence_mean_rmse, on_step=False, on_epoch=True)

        self.log('val/arousal_std_rmse', arousal_std_rmse, on_step=False, on_epoch=True)
        self.log('val/valence_std_rmse', valence_std_rmse, on_step=False, on_epoch=True)