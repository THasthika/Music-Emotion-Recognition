import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import torchmetrics as tm

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from utils.common import Conv2DBlock, Conv1DBlock, LinearBlock

from models.base import BaseModel

class Model1(BaseModel):

    def __init__(self, batch_size, num_workers, config):
        super().__init__(batch_size, num_workers, dataset_class=None, dataset_class_args=None, split_dir=None)
        self.config = config
        self.lr = self.__set_model_parameter(config, 'lr', 0.01)
        self.raw_audio_extractor_units = self.__set_model_parameter(
            config,
            ['raw_audio_extractor_units'],
            [1, 64, 128]
        )
        self.raw_audio_latent_time_units = self.__set_model_parameter(
            config,
            ['raw_audio_latent_time_units'],
            128
        )
        self.classifier_units = self.__set_model_parameter(
            config,
            ['classifier_units'],
            [1024, 512, 128]
        )

        ## build layers
        self.__build()

        ## metrics
        self.train_acc = tm.Accuracy(top_k=3)
        self.val_acc = tm.Accuracy(top_k=3)

        self.test_acc = tm.Accuracy(top_k=3)
        self.test_f1_class = tm.F1(num_classes=4, average='none')
        self.test_f1_global = tm.F1(num_classes=4)

        ## loss
        self.cat_loss = F.cross_entropy

    def __build(self):

        # 0 -> 1, 661500

        self.raw_audio_feature_extractor = self.__create_conv_network(
            self.raw_audio_extractor_units,
            self.raw_audio_latent_time_units
        )

        input_size = self.raw_audio_extractor_units[-1] * self.raw_audio_latent_time_units

        self.classifier_units = [
            input_size,
            *self.classifier_units
        ]

        self.classifier = self.__create_linear_network(self.classifier_units)


    def forward(self, x):

        raw_audio = x['raw']

        x = self.raw_audio_feature_extractor(raw_audio)
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