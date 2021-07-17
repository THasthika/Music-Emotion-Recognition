from re import M
from models import BaseCatModel
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nnAudio import Spectrogram
import torchmetrics as tm

class AC2DConvCat_V2(BaseCatModel):

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


        self.audio_feature_1d_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=500, kernel_size=1024, stride=256),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(num_features=500),
            nn.ReLU(),
        )

        self.audio_feature_2d_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=256),
            nn.Dropout2d(p=self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(output_size=(60, 100))
        )

        self.stft_feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=256),
            nn.Dropout2d(p=self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(output_size=(62, 100))
        )

        self.mel_spec_feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=256),
            nn.Dropout2d(p=self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(output_size=(14, 100))
        )

        self.mfcc_feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.BatchNorm2d(num_features=256),
            nn.Dropout2d(p=self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(output_size=(14, 100))
        )

        self.mid_extractor = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5), stride=(2, 2)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(5, 5), stride=(2, 2)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=512),
            nn.Dropout2d(p=self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(5, 3), stride=(2, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=1024),
            nn.Dropout2d(p=self.config[self.DROPOUT]),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=512),
            nn.Dropout(p=self.config[self.DROPOUT]),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Dropout(p=self.config[self.DROPOUT]),
            nn.Linear(in_features=128, out_features=4)
        )

    def forward(self, x):
        
        audio_x = self.audio_feature_1d_extractor(x)
        audio_x = torch.unsqueeze(audio_x, dim=1)
        audio_x = self.audio_feature_2d_extractor(audio_x)

        stft_x = self.stft(x)
        stft_x = torch.unsqueeze(stft_x, dim=1)
        stft_x = self.stft_feature_extractor(stft_x)

        mel_x = self.mel_spec(x)
        mel_x = torch.unsqueeze(mel_x, dim=1)
        mel_x = self.mel_spec_feature_extractor(mel_x)

        mfcc_x = self.mfcc(x)
        mfcc_x = torch.unsqueeze(mfcc_x, dim=1)
        mfcc_x = self.mfcc_feature_extractor(mfcc_x)

        x = torch.cat((audio_x, stft_x, mel_x, mfcc_x), dim=2)

        x = self.mid_extractor(x)
        x = torch.flatten(x, start_dim=1)

        x = self.fc(x)
        return x
