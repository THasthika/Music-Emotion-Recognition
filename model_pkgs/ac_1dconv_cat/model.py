import numpy as np
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchmetrics as tm

class AC1DConvCat(pl.LightningModule):

    LR = "lr"
    AUDIO_ADAPTIVE_LAYER_UNITS = "audio_adaptive_layer_units"
    COMPUTED_ADAPTIVE_LAYER_UNITS = "computed_adaptive_layer_units"

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

        self.__build_model()

        ## metrics
        self.train_acc = tm.Accuracy(top_k=3)
        self.val_acc = tm.Accuracy(top_k=3)

        self.test_acc = tm.Accuracy(top_k=3)
        self.test_f1_class = tm.F1(num_classes=4, average='none')
        self.test_f1_global = tm.F1(num_classes=4)

        ## loss
        self.loss = F.cross_entropy
    
    def __build_model(self):

        self.audio_feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=250, kernel_size=1024, stride=256),
            nn.BatchNorm1d(250),
            nn.Dropout(),
            nn.ReLU(),

            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=13, stride=5),
            nn.BatchNorm1d(250),
            nn.Dropout(),
            nn.ReLU(),

            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=13, stride=5),
            nn.BatchNorm1d(250),
            nn.Dropout(),
            nn.ReLU(),

            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=13, stride=5),
            nn.BatchNorm1d(250),
            nn.Dropout(),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(output_size=self.config[self.AUDIO_ADAPTIVE_LAYER_UNITS]),
            nn.Dropout()
        )

        self.computed_feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=692, out_channels=500, kernel_size=7, stride=3),
            nn.BatchNorm1d(500),
            nn.Dropout(),
            nn.ReLU(),

            nn.Conv1d(in_channels=500, out_channels=500, kernel_size=7, stride=3),
            nn.BatchNorm1d(500),
            nn.Dropout(),
            nn.ReLU(),

            nn.Conv1d(in_channels=500, out_channels=500, kernel_size=7, stride=3),
            nn.BatchNorm1d(500),
            nn.Dropout(),
            nn.ReLU(),

            nn.Conv1d(in_channels=500, out_channels=500, kernel_size=7, stride=3),
            nn.BatchNorm1d(500),
            nn.Dropout(),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(output_size=self.config[self.COMPUTED_ADAPTIVE_LAYER_UNITS]),
            nn.Dropout()
        )

        input_size = self.config[self.AUDIO_ADAPTIVE_LAYER_UNITS] * 250
        input_size += self.config[self.COMPUTED_ADAPTIVE_LAYER_UNITS] * 500

        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=4)
        )

    def forward(self, x):
        
        audio_x = x['audio']
        computed_x = torch.column_stack([
            x['spec'],
            x['mel_spec'],
            x['mfccs'],
            x['chroma'],
            x['tonnetz'],
            x['spectral_contrast'],
            x['spectral_aggregate']
        ])

        audio_x = self.audio_feature_extractor(audio_x)
        computed_x = self.computed_feature_extractor(computed_x)

        audio_x = torch.flatten(audio_x, start_dim=1)
        computed_x = torch.flatten(computed_x, start_dim=1)

        print(audio_x.shape)
        print(computed_x.shape)

        x = torch.cat((audio_x, computed_x), dim=1)

        print(x.shape)

        x = self.classifier(x)
        return x

    def predict(self, x):
        x = self.forward(x)
        return F.softmax(x, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config[self.LR])
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

    def train_dataloader(self):
        if self.test_ds is None: return None
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        if self.val_ds is None: return None
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        if self.test_ds is None: return None
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)