import torch
from torch.functional import stft
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import dropout
from torch.utils.data import DataLoader

from collections import OrderedDict

import torchmetrics as tm

from nnAudio import Spectrogram

from models import BaseCatModel
from utils.layer import Unsqueeze

"""
1D + 2D Model
Multi Scaled Approach

512 / 128
1024 / 256
11025 / 2756
"""


class H1DConvCat_V1(BaseCatModel):

    def __init__(self,
                batch_size=32,
                num_workers=4,
                train_ds=None,
                val_ds=None,
                test_ds=None,
                **model_config):
        super().__init__(batch_size, num_workers, train_ds, val_ds, test_ds, **model_config)
        self.__build_model()

    def __make_1d_block(self, in_channels, out_channels, kernel_size, stride, dropout=0.5, batch_normalize=True, activation="relu", max_pool=3, max_pool_stride=1):
        l = []
        # add the conv1 layer
        l.append(("conv1", nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)))
        if not dropout is None:
            l.append(("dropout", nn.Dropout(p=dropout)))
        if batch_normalize:
            l.append(("bn", nn.BatchNorm1d(out_channels)))
        if activation == "relu":
            l.append(("act", nn.ReLU()))
        if not max_pool is None:
            l.append(("max_pool", nn.MaxPool1d(max_pool, stride=max_pool_stride)))
        return nn.Sequential(OrderedDict(l))

    def __make_2d_block(self, in_channels, out_channels, kernel_size, stride, dropout=0.5, batch_normalize=True, activation="relu", max_pool=(3, 3), max_pool_stride=(1, 1)):
        l = []
        # add the conv2 layer
        l.append(("conv2", nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)))
        if not dropout is None:
            l.append(("dropout", nn.Dropout2d(p=dropout)))
        if batch_normalize:
            l.append(("bn", nn.BatchNorm2d(out_channels)))
        if activation == "relu":
            l.append(("act", nn.ReLU()))
        if not max_pool is None:
            l.append(("max_pool", nn.MaxPool2d(max_pool, stride=max_pool_stride)))
        return nn.Sequential(OrderedDict(l))

    def __make_linear(self, in_features, out_features, dropout=0.5, batch_normalize=True, activation="relu"):
        l = []
        l.append(("linear", nn.Linear(in_features=in_features, out_features=out_features)))
        if not dropout is None:
            l.append(("dropout", nn.Dropout(p=dropout)))
        if batch_normalize:
            l.append(("bn", nn.BatchNorm1d(out_features)))
        if activation == "relu":
            l.append(("act", nn.ReLU()))
        return nn.Sequential(OrderedDict(l))

    def __build_model(self):

        stft_nn = []

        stft_nn_0 = nn.Sequential(
            Spectrogram.STFT(n_fft=1024, fmax=9000, trainable=False, output_format="Magnitude"),
            
            Unsqueeze(1),

            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(13, 5), stride=(2, 2)),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(13, 5), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(7, 3), stride=(1, 1)),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(7, 5), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 5), stride=(1, 1)),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 5), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(6, 5), stride=(1, 1)),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(output_size=(1, 1)),

            nn.Flatten(start_dim=1),

            nn.Linear(in_features=128, out_features=64),
            nn.Dropout(p=self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.Linear(in_features=64, out_features=64),
            nn.Dropout(p=self.config[self.DROPOUT]),
            nn.ReLU()
        )

        stft_nn.append(stft_nn_0)

        # stft_nn_1 = nn.Sequential(
        #     Spectrogram.STFT(n_fft=2048, fmax=9000, trainable=False, output_format="Magnitude"),

        #     Unsqueeze(1),

        #     nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(7, 3), stride=(2, 1)),
        #     nn.ReLU(),
            
        #     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(7, 3), stride=(2, 1)),
        #     nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
        #     nn.ReLU(),

        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(7, 3), stride=(1, 1)),
        #     nn.ReLU(),
            
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(7, 3), stride=(1, 1)),
        #     nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
        #     nn.ReLU(),

        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 3), stride=(1, 1)),
        #     nn.ReLU(),
            
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 3), stride=(1, 1)),
        #     nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
        #     nn.ReLU(),

        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        #     nn.ReLU(),
            
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        #     nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
        #     nn.ReLU(),

        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        #     nn.ReLU(),
            
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        #     nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
        #     nn.ReLU(),

        #     nn.Flatten(start_dim=1),

        #     nn.Linear(in_features=128, out_features=64),
        #     nn.Dropout(p=self.config[self.DROPOUT]),
        #     nn.ReLU(),

        #     nn.Linear(in_features=64, out_features=64),
        #     nn.Dropout(p=self.config[self.DROPOUT]),
        #     nn.ReLU(),
        # )

        # stft_nn.append(stft_nn_1)

        # stft_nn_2 = nn.Sequential(
        #     Spectrogram.STFT(n_fft=4096, fmax=9000, trainable=False, output_format="Magnitude"),

        #     Unsqueeze(1),
            
        #     nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(7, 3), stride=(3, 1)),
        #     nn.ReLU(),
            
        #     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(7, 3), stride=(3, 1)),
        #     nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
        #     nn.ReLU(),

        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(7, 3), stride=(2, 1)),
        #     nn.ReLU(),
            
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(7, 3), stride=(1, 1)),
        #     nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
        #     nn.ReLU(),

        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(1, 1)),
        #     nn.ReLU(),
            
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        #     nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
        #     nn.ReLU(),

        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        #     nn.ReLU(),
            
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        #     nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
        #     nn.ReLU(),

        #     nn.Flatten(start_dim=1),

        #     nn.Linear(in_features=128, out_features=64),
        #     nn.Dropout(p=self.config[self.DROPOUT]),
        #     nn.ReLU(),

        #     nn.Linear(in_features=64, out_features=64),
        #     nn.Dropout(p=self.config[self.DROPOUT]),
        #     nn.ReLU(),
        # )

        # stft_nn.append(stft_nn_2)

        self.stft_nn = nn.ModuleList(stft_nn)

        input_size = 64 * 1

        self.fc = nn.Sequential(

            nn.Linear(in_features=input_size, out_features=128),
            nn.Dropout(p=self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=128),
            nn.Dropout(p=self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=64),
            nn.Dropout(p=self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.Linear(in_features=64, out_features=4),
            nn.Dropout(p=self.config[self.DROPOUT]),
            nn.ReLU()
        )

    def forward(self, x):
        stft_x = list(map(lambda net: net(x), self.stft_nn))
        x = torch.cat(stft_x, dim=1)
        x = self.fc(x)
        return x
