import pytorch_lightning as pl

from models.base import WandbBaseModel

import torch
from torch import nn
from torch.nn import functional as F

import torchmetrics as tm

"""
ModelCatA - A 1D convolutional Model

Conv1
Conv2
Conv3
GlobalAvgPool
FullyConnected1
FullyConnected2
Softmax

"""

class ModelCatA(WandbBaseModel):

    CMDS = [
        ('lr', float, 0.001),
        ('adaptive_layer', int, 128),
        ('max_epochs', int, 100),
        ('conv1_kernel_size', int, 11025),
        ('conv1_kernel_stride', int, 400)
    ]

    def __init__(self,
                batch_size=32,
                num_workers=4,
                sample_rate=22050,
                chunk_duration=5,
                overlap=2.5,
                data_artifact=None,
                split_artifact=None,
                **config):
        """[summary]

        Args:
            batch_size (int, optional): [description]. Defaults to 32.
            num_workers (int, optional): [description]. Defaults to 4.
            data_artifact ([type], optional): [description]. Defaults to None.
            split_artifact ([type], optional): [description]. Defaults to None.

        Config:
            lr
            conv1_kernel_size
            conv1_kernel_stride
            adaptive_layer
        """
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            overlap=overlap,
            data_artifact=data_artifact,
            split_artifact=split_artifact,
            label_type="categorical")
        self.config = config
        self.lr = config['lr']

        ## build layers
        self.__build()

        ## metrics
        self.train_acc = tm.Accuracy(top_k=3)
        self.val_acc = tm.Accuracy(top_k=3)

        self.test_acc = tm.Accuracy(top_k=3)
        self.test_f1_class = tm.F1(num_classes=4, average='none')
        self.test_f1_global = tm.F1(num_classes=4)

        ## loss
        self.loss = F.cross_entropy

    def __build(self):

        channel_count = 250
        kernel_size = 7
        stride = 1

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=channel_count, kernel_size=self.config['conv1_kernel_size'], stride=self.config['conv1_kernel_stride']),
            nn.BatchNorm1d(channel_count),
            nn.ReLU(),

            nn.Conv1d(in_channels=channel_count, out_channels=channel_count, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm1d(channel_count),
            nn.ReLU(),

            nn.Conv1d(in_channels=channel_count, out_channels=channel_count, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm1d(channel_count),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(output_size=self.config['adaptive_layer']),
            nn.BatchNorm1d(channel_count),
            nn.Dropout()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.config['adaptive_layer']*channel_count, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=4)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def predict(self, x):
        x = self.forward(x)
        return F.softmax(x, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

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