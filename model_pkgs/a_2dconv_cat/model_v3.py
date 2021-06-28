import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchmetrics as tm

from nnAudio import Spectrogram as nnSpectrogram

class A2DConvCat_V1(pl.LightningModule):

    LR = "lr"
    ADAPTIVE_LAYER_UNITS_0 = "adaptive_layer_units_0"
    ADAPTIVE_LAYER_UNITS_1 = "adaptive_layer_units_1"
    N_FFT = "n_fft"
    N_MELS = "n_mels"
    N_MFCC = "n_mfcc"
    SPEC_TRAINABLE = "spec_trainable"

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
        self.val_f1_class = tm.F1(num_classes=4, average='none')
        self.val_f1_global = tm.F1(num_classes=4)

        self.test_acc = tm.Accuracy(top_k=3)
        self.test_f1_class = tm.F1(num_classes=4, average='none')
        self.test_f1_global = tm.F1(num_classes=4)

        ## loss
        self.loss = F.cross_entropy
    
    def __build_model(self):

        f_bins = (self.config[self.N_FFT] // 2) + 1

        self.stft = nnSpectrogram.STFT(n_fft=self.config[self.N_FFT], fmax=9000, sr=22050, trainable=self.config[self.SPEC_TRAINABLE], output_format="Magnitude")
        self.mel_spec = nnSpectrogram.MelSpectrogram(sr=22050, n_fft=self.config[self.N_FFT], n_mels=self.config[self.N_MELS], trainable_mel=self.config[self.SPEC_TRAINABLE], trainable_STFT=self.config[self.SPEC_TRAINABLE])
        self.mfccs = nnSpectrogram.MFCC(sr=22050, n_mfcc=self.config[self.N_MFCC])

        self.stft_feature_extractor = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=256),
            nn.Dropout2d(),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=256),
            nn.Dropout2d(),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(output_size=(
                self.config[self.ADAPTIVE_LAYER_UNITS_0],
                self.config[self.ADAPTIVE_LAYER_UNITS_1]
            ))
        )

        self.mel_spec_feature_extractor = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=128),
            nn.Dropout2d(),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=256),
            nn.Dropout2d(),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(output_size=(
                self.config[self.ADAPTIVE_LAYER_UNITS_0],
                self.config[self.ADAPTIVE_LAYER_UNITS_1]
            ))
        )

        self.mfcc_extractor_feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(output_size=(
                self.config[self.ADAPTIVE_LAYER_UNITS_0],
                self.config[self.ADAPTIVE_LAYER_UNITS_1]
            ))
        )

        out_channels = 256
        input_size = (self.config[self.ADAPTIVE_LAYER_UNITS_0] * self.config[self.ADAPTIVE_LAYER_UNITS_1] * out_channels)

        out_channels = 256
        input_size += (self.config[self.ADAPTIVE_LAYER_UNITS_0] * self.config[self.ADAPTIVE_LAYER_UNITS_1] * out_channels)

        out_channels = 16
        input_size += (self.config[self.ADAPTIVE_LAYER_UNITS_0] * self.config[self.ADAPTIVE_LAYER_UNITS_1] * out_channels)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=4)
        )

    def forward(self, x):
        x = self.stft(x)
        x = torch.unsqueeze(x, dim=1)
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
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

    def train_dataloader(self):
        if self.test_ds is None: return None
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        if self.val_ds is None: return None
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def test_dataloader(self):
        if self.test_ds is None: return None
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)