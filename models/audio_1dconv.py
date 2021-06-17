import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import torchmetrics as tm

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from utils.common import Conv2DBlock, Conv1DBlock, LinearBlock

from models.base import BaseModel

class Audio1DConv(BaseModel):

    def __init__(self, batch_size, num_workers, config):
        super().__init__(batch_size, num_workers, dataset_class=None, dataset_class_args=None, split_dir=None)
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
        self.classifier_units = self.set_model_parameter(
            config,
            ['classifier_units'],
            [1024, 512, 128, { 'features': 4, }]
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
        self.reg_loss = F.l1_loss

    def __build(self):

        # 0 -> 1, 661500

        self.raw_audio_feature_extractor = self.create_conv_network(
            self.raw_audio_extractor_units,
            self.raw_audio_latent_time_units
        )

        input_size = self.raw_audio_extractor_units[-1] * self.raw_audio_latent_time_units

        self.mid_representation = nn.Sequential(
            LinearBlock(in_features=input_size, out_features=1024),
            LinearBlock(in_features=1024, out_features=512)
        )

        self.class_out = nn.Sequential(
            LinearBlock(in_features=512, out_features=128),
            LinearBlock(in_features=128, out_features=4, dropout=False, activation=None)
        )

        self.av_out = nn.Sequential(
            LinearBlock(in_features=512, out_features=128),
            LinearBlock(in_features=128, out_features=4, dropout=False, activation=None)
        )


    def forward(self, x):

        raw_audio = x

        x = self.raw_audio_feature_extractor(raw_audio)
        x = torch.flatten(x, start_dim=1)

        x = self.mid_representation(x)

        cls_out = self.class_out(x)
        av_out = self.av_out(x)

        return (cls_out, av_out)

    def predict(self, x):
        x = self.forward(x)
        return F.softmax(x, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, (cat_y, reg_y) = batch

        (y_logit, y_regress) = self(x)
        cat_loss = self.cat_loss(y_logit, cat_y)
        pred = F.softmax(y_logit, dim=1)

        reg_loss = self.reg_loss(y_regress, reg_y)

        self.log('train/cat_loss', cat_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train/reg_loss', reg_loss, prog_bar=True, on_step=False, on_epoch=True)

        self.log('train/acc', self.train_acc(pred, cat_y), prog_bar=True, on_step=False, on_epoch=True)

        return cat_loss

    def validation_step(self, batch, batch_idx):
        x, (cat_y, reg_y) = batch

        (y_logit, y_regress) = self(x)
        cat_loss = self.cat_loss(y_logit, cat_y)
        pred = F.softmax(y_logit, dim=1)

        reg_loss = self.reg_loss(y_regress, reg_y)
        
        self.log('val/cat_loss', cat_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/reg_loss', reg_loss, prog_bar=True, on_step=False, on_epoch=True)

        self.log("val/acc", self.val_acc(pred, cat_y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, (cat_y, reg_y) = batch

        (y_logit, y_regress) = self(x)
        
        cat_loss = self.cat_loss(y_logit, cat_y)
        pred = F.softmax(y_logit, dim=1)

        reg_loss = self.reg_loss(y_regress, reg_y)

        self.log('test/cat_loss', cat_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test/reg_loss', reg_loss, prog_bar=True, on_step=False, on_epoch=True)

        self.log("test/acc", self.val_acc(pred, cat_y), prog_bar=True)
        self.log("test/f1_global", self.test_f1_global(pred, cat_y))

        f1_scores = self.test_f1_class(pred, cat_y)
        for (i, x) in enumerate(torch.flatten(f1_scores)):
            self.log("test/f1_class_{}".format(i), x)