import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import torchmetrics as tm

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from utils.common import Conv2DBlock, Conv1DBlock, LinearBlock

from models.base import BaseModel

class AudioComputed1DConvCat(BaseModel):

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

        self.raw_audio_linear_units = self.set_model_parameter(
            config,
            ['raw_audio_linear_units'],
            [1024, 512]
        )

        self.computed_feature_units = self.set_model_parameter(
            config,
            ['computed_feature_units'],
            [692, 64]
        )

        self.computed_feature_time_units = self.set_model_parameter(
            config,
            ['computed_feature_time_units'],
            128
        )

        self.computed_linear_units = self.set_model_parameter(
            config,
            ['computed_linear_units'],
            [1024, 512]
        )
        
        self.classifier_units = self.set_model_parameter(
            config,
            ['classifier_units'],
            [1024, 512, 128]
        )

        self.n_classes = self.set_model_parameter(
            config,
            ['n_classes'],
            4
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
        self.loss = F.cross_entropy

    def __build(self):

        # 0 -> 1, 110250
        # 1 -> 692, 216

        self.raw_audio_feature_extractor = self.create_conv_network(
            self.raw_audio_extractor_units,
            self.raw_audio_latent_time_units
        )

        raw_feature_size = self.raw_audio_extractor_units[-1]['channels'] if type(self.raw_audio_extractor_units[-1]) is dict else self.raw_audio_extractor_units[-1]

        self.raw_audio_linear_net = self.create_linear_network([
            raw_feature_size * self.raw_audio_latent_time_units,
            *self.raw_audio_linear_units
        ])

        self.computed_feature_extractor = self.create_conv_network(
            self.computed_feature_units,
            self.computed_feature_time_units
        )

        calcaulted_feature_size = self.computed_feature_units[-1]['channels'] if type(self.computed_feature_units[-1]) is dict else self.computed_feature_units[-1]

        self.computed_linear_net = self.create_linear_network([
            calcaulted_feature_size * self.computed_feature_time_units,
            *self.computed_linear_units
        ])

        raw_feature_size = self.raw_audio_linear_units[-1]['features'] if type(self.raw_audio_linear_units[-1]) is dict else self.raw_audio_linear_units[-1]
        calcaulted_feature_size = self.computed_linear_units[-1]['features'] if type(self.computed_linear_units[-1]) is dict else self.computed_linear_units[-1]

        input_size = raw_feature_size + calcaulted_feature_size

        self.classifier_units = [
            input_size,
            *self.classifier_units,
            { 'features': self.n_classes, 'activation': None, 'dropout': False, 'batch_normalize': False }
        ]

        self.classifier = self.create_linear_network(self.classifier_units)


    def forward(self, x):

        raw_audio = x['audio']

        concat_keys = [
            'spec',
            'mel_spec',
            'mfccs',
            'chroma',
            'tonnetz',
            'spectral_contrast',
            'spectral_aggregate'
        ]

        t = tuple(map(lambda k: x[k], concat_keys))
        farr = torch.cat(t, dim=1)

        x0 = self.raw_audio_feature_extractor(raw_audio)
        x0 = torch.flatten(x0, start_dim=1)
        x0 = self.raw_audio_linear_net(x0)

        x1 = self.computed_feature_extractor(farr)
        x1 = torch.flatten(x1, start_dim=1)
        x1 = self.computed_linear_net(x1)

        x = torch.cat((x0, x1), dim=1)

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