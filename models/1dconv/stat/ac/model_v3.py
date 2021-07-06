import numpy as np
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchmetrics as tm

from nnAudio import Spectrogram

from utils.loss import rmse_loss


class AC1DConvStat_V2(pl.LightningModule):

    LR = "lr"
    ADAPTIVE_LAYER_UNITS = "adaptive_layer_units"

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

        ## loss
        self.loss = rmse_loss

        self.test_arousal_r2 = tm.R2Score(num_outputs=1)
        self.test_valence_r2 = tm.R2Score(num_outputs=1)
        
        self.test_r2score = tm.R2Score(num_outputs=2)
    
    def __build_model(self):

        f_bins = (self.config[self.N_FFT] // 2) + 1

        self.stft = Spectrogram.STFT(n_fft=self.config[self.N_FFT], fmax=9000, sr=22050, trainable=self.config[self.SPEC_TRAINABLE], output_format="Magnitude")
        self.mel_spec = Spectrogram.MelSpectrogram(sr=22050, n_fft=self.config[self.N_FFT], n_mels=self.config[self.N_MELS], trainable_mel=self.config[self.SPEC_TRAINABLE], trainable_STFT=self.config[self.SPEC_TRAINABLE])
        self.mfcc = Spectrogram.MFCC(sr=22050, n_mfcc=self.config[self.N_MFCC])

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

            nn.AdaptiveAvgPool1d(output_size=self.config[self.ADAPTIVE_LAYER_UNITS]),
            nn.Dropout()
        )

        self.stft_feature_extractor = nn.Sequential(

            nn.Conv1d(in_channels=f_bins, out_channels=500, kernel_size=3, stride=1),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(num_features=500),
            nn.ReLU(),

            nn.Conv1d(in_channels=500, out_channels=500, kernel_size=3, stride=1),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(num_features=500),
            nn.ReLU(),

            nn.Conv1d(in_channels=500, out_channels=500, kernel_size=3, stride=1),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(num_features=500),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(output_size=self.config[self.ADAPTIVE_LAYER_UNITS])
        )

        self.mel_spec_feature_extractor = nn.Sequential(

            nn.Conv1d(in_channels=self.config[self.N_MELS], out_channels=100, kernel_size=3, stride=1),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(num_features=100),
            nn.ReLU(),

            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3, stride=1),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(num_features=100),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(output_size=self.config[self.ADAPTIVE_LAYER_UNITS])
        )

        self.mfcc_feature_extractor = nn.Sequential(

            nn.Conv1d(in_channels=self.config[self.N_MFCC], out_channels=16, kernel_size=3, stride=1),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),

            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),

            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(output_size=self.config[self.ADAPTIVE_LAYER_UNITS])
        )

        out_channels = 250
        input_size = (self.config[self.ADAPTIVE_LAYER_UNITS] * out_channels)

        out_channels = 500
        input_size += (self.config[self.ADAPTIVE_LAYER_UNITS] * out_channels)

        out_channels = 100
        input_size += (self.config[self.ADAPTIVE_LAYER_UNITS] * out_channels)

        out_channels = 16
        input_size += (self.config[self.ADAPTIVE_LAYER_UNITS] * out_channels)

        self.fc = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=512),
            nn.Dropout(),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=2)
        )

    def forward(self, x):

        audio_x = self.audio_feature_extractor(x)

        stft_x = self.stft(x)
        stft_x = self.stft_feature_extractor(stft_x)

        mel_x = self.mel_spec(x)
        mel_x = self.mel_spec_feature_extractor(mel_x)

        mfcc_x = self.mfcc(x)
        mfcc_x = self.mfcc_feature_extractor(mfcc_x)

        audio_x = torch.flatten(audio_x, start_dim=1)
        stft_x = torch.flatten(stft_x, start_dim=1)
        mel_x = torch.flatten(mel_x, start_dim=1)
        mfcc_x = torch.flatten(mfcc_x, start_dim=1)

        x = torch.cat((audio_x, stft_x, mel_x, mfcc_x), dim=1)

        x = self.fc(x)
        
        return x

    def predict(self, x):
        x = self.forward(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config[self.LR])
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch

        pred = self(x)
        loss = self.loss(pred, y[:, [0, 1]])

        self.log('train/loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        pred = self(x)
        loss = self.loss(pred, y[:, [0, 1]])

        arousal_rmse = self.loss(pred[:, 1], y[:, 1])
        valence_rmse = self.loss(pred[:, 0], y[:, 0])

        self.log("val/loss", loss, prog_bar=True)

        self.log("val/arousal_rmse", arousal_rmse, on_step=False, on_epoch=True)
        self.log("val/valence_rmse", valence_rmse, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch

        pred = self(x)
        loss = self.loss(pred, y[:, [0, 1]])

        r2score = self.test_r2score(pred, y[:, [0, 1]])
        arousal_r2score = self.test_arousal_r2(pred[:, 1], y[:, 1])
        valence_r2score = self.test_valence_r2(pred[:, 0], y[:, 0])

        arousal_rmse = self.loss(pred[:, 1], y[:, 1])
        valence_rmse = self.loss(pred[:, 0], y[:, 0])

        self.log("test/loss", loss)

        self.log('test/r2score', r2score, on_step=False, on_epoch=True)
        self.log('test/arousal_r2score', arousal_r2score, on_step=False, on_epoch=True)
        self.log('test/valence_r2score', valence_r2score, on_step=False, on_epoch=True)

        self.log("test/arousal_rmse", arousal_rmse, on_step=False, on_epoch=True)
        self.log("test/valence_rmse", valence_rmse, on_step=False, on_epoch=True)

    def train_dataloader(self):
        if self.test_ds is None: return None
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        if self.val_ds is None: return None
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def test_dataloader(self):
        if self.test_ds is None: return None
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)