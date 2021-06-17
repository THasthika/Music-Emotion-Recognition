import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import torchmetrics as tm

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from utils.common import Conv2DBlock, Conv1DBlock, LinearBlock

from models.base import BaseModel

class Audio1DConvStat(BaseModel):

    def __init__(self, **config):
        super().__init__(batch_size=32, num_workers=2, dataset_class=None, dataset_class_args=None, split_dir=None)

        self.config = config

        self.lr = self.set_model_parameter(config, 'lr', 0.01)

        self.raw_audio_extractor_units = self.set_model_parameter(
            config,
            ['raw_audio_extractor_units'],
            [1, 64, 128]
        )

        self.raw_audio_latent_time_units = self.set_model_parameter(
            config,
            ['raw_audio_latent_time_units'],
            128
        )
        
        self.regressor_units = self.set_model_parameter(
            config,
            ['regressor_units'],
            [1024, 512, 128]
        )

        ## build layers
        self.__build()

        ## loss
        self.loss = F.l1_loss

    def __build(self):

        # 0 -> 1, 22050 * n

        self.raw_audio_feature_extractor = self.create_conv_network(
            self.raw_audio_extractor_units,
            self.raw_audio_latent_time_units
        )

        input_size = self.raw_audio_extractor_units[-1] * self.raw_audio_latent_time_units

        self.regressor_units = [
            input_size,
            *self.regressor_units,
            { 'features': 4, 'activation': None, 'dropout': False, 'batch_normalize': False }
        ]

        self.regressor = self.create_linear_network(self.regressor_units)


    def forward(self, x):

        raw_audio = x

        x = self.raw_audio_feature_extractor(raw_audio)
        x = torch.flatten(x, start_dim=1)

        x = self.regressor(x)

        return x

    def predict(self, x):
        x = self.forward(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_h = self(x)
        loss = self.loss(y_h, y)

        self.log('train/loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_h = self(x)
        loss = self.loss(y_h, y)
        
        self.log("val/loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_h = self(x)
        loss = self.loss(y_h, y)

        self.log("test/loss", loss)