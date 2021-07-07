from models.base import BaseStatModel
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nnAudio import Spectrogram
import torchmetrics as tm
from utils.activation import CustomELU

class C2DConvStat_V1(BaseStatModel):

    ADAPTIVE_LAYER_UNITS_0 = "adaptive_layer_units_0"
    ADAPTIVE_LAYER_UNITS_1 = "adaptive_layer_units_1"
    N_FFT = "n_fft"
    N_MELS = "n_mels"
    N_MFCC = "n_mfcc"
    SPEC_TRAINABLE = "spec_trainable"

    def __init__(self,
                batch_size=32,
                num_workers=4,
                train_ds=None,
                val_ds=None,
                test_ds=None,
                **model_config):
        super().__init__(batch_size, num_workers, train_ds, val_ds, test_ds, **model_config)

        self.__build_model()
    
    def __build_model(self):

        f_bins = (self.config[self.N_FFT] // 2) + 1

        self.stft = Spectrogram.STFT(n_fft=self.config[self.N_FFT], fmax=9000, sr=22050, trainable=self.config[self.SPEC_TRAINABLE], output_format="Magnitude")
        self.mel_spec = Spectrogram.MelSpectrogram(sr=22050, n_fft=self.config[self.N_FFT], n_mels=self.config[self.N_MELS], trainable_mel=self.config[self.SPEC_TRAINABLE], trainable_STFT=self.config[self.SPEC_TRAINABLE])
        self.mfcc = Spectrogram.MFCC(sr=22050, n_mfcc=self.config[self.N_MFCC])

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
            nn.Dropout2d(p=self.config[self.DROPOUT]),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=256),
            nn.Dropout2d(p=self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(output_size=(
                self.config[self.ADAPTIVE_LAYER_UNITS_0],
                self.config[self.ADAPTIVE_LAYER_UNITS_1]
            ))
        )

        out_channels = 256
        input_size = (self.config[self.ADAPTIVE_LAYER_UNITS_0] * self.config[self.ADAPTIVE_LAYER_UNITS_1] * out_channels)

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
            nn.Dropout2d(p=self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=256),
            nn.Dropout2d(p=self.config[self.DROPOUT]),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d(output_size=(
                self.config[self.ADAPTIVE_LAYER_UNITS_0],
                self.config[self.ADAPTIVE_LAYER_UNITS_1]
            ))
        )

        out_channels = 256
        input_size += (self.config[self.ADAPTIVE_LAYER_UNITS_0] * self.config[self.ADAPTIVE_LAYER_UNITS_1] * out_channels)

        self.mfcc_feature_extractor = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 8)),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 4)),
            nn.BatchNorm2d(num_features=32),
            nn.Dropout2d(p=self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(num_features=64),
            nn.Dropout2d(p=self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(output_size=(
                self.config[self.ADAPTIVE_LAYER_UNITS_0],
                self.config[self.ADAPTIVE_LAYER_UNITS_1]
            ))
        )

        out_channels = 64
        input_size += (self.config[self.ADAPTIVE_LAYER_UNITS_0] * self.config[self.ADAPTIVE_LAYER_UNITS_1] * out_channels)

        self.fc0 = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=512),
            nn.Dropout(p=self.config[self.DROPOUT]),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU()
        )

        self.fc_mean = nn.Sequential(
            nn.Linear(in_features=128, out_features=2)
        )

        self.fc_std = nn.Sequential(
            nn.Linear(in_features=128, out_features=2),
            self._get_std_activation()
        )

    def forward(self, x):
        
        stft_x = self.stft(x)
        stft_x = torch.unsqueeze(stft_x, dim=1)
        stft_x = self.stft_feature_extractor(stft_x)

        mel_x = self.mel_spec(x)
        mel_x = torch.unsqueeze(mel_x, dim=1)
        mel_x = self.mel_spec_feature_extractor(mel_x)

        mfcc_x = self.mfcc(x)
        mfcc_x = torch.unsqueeze(mfcc_x, dim=1)
        mfcc_x = self.mfcc_feature_extractor(mfcc_x)

        stft_x = torch.flatten(stft_x, start_dim=1)
        mel_x = torch.flatten(mel_x, start_dim=1)
        mfcc_x = torch.flatten(mfcc_x, start_dim=1)

        x = torch.cat((stft_x, mel_x, mfcc_x), dim=1)

        x = self.fc0(x)

        x_mean = self.fc_mean(x)
        x_std = self.fc_std(x)
        x = torch.cat((x_mean, x_std), dim=1)
        return x

    def predict(self, x):
        x = self.forward(x)
        return F.softmax(x, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config[self.LR])
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch

        pred = self(x)
        loss = self.loss(pred, y)
        distanceMeasure = self.train_distance(pred, y)

        self.log('train/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train/distance', distanceMeasure, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        pred = self(x)
        loss = self.loss(pred, y)
        distanceMeasure = self.val_distance(pred, y)

        self.log("val/loss", loss, prog_bar=True)
        self.log('val/distance', distanceMeasure, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch

        pred = self(x)
        loss = self.loss(pred, y)
        distanceMeasure = self.test_distance(pred, y)
        r2score = self.test_r2score(pred, y)

        self.log("test/loss", loss)
        self.log('test/distance', distanceMeasure)

        self.log('test/r2score', r2score, on_step=False, on_epoch=True)

    def train_dataloader(self):
        if self.test_ds is None: return None
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        if self.val_ds is None: return None
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def test_dataloader(self):
        if self.test_ds is None: return None
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)