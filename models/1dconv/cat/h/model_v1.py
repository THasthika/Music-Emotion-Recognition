import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from collections import OrderedDict

import torchmetrics as tm

from models import BaseCatModel

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

        self.feature_extractor_0 = nn.Sequential(
            self.__make_1d_block(in_channels=1, out_channels=25, kernel_size=512, stride=128, dropout=self.config[self.DROPOUT]),
            self.__make_1d_block(in_channels=25, out_channels=50, kernel_size=7, stride=2, dropout=self.config[self.DROPOUT]),
            self.__make_1d_block(in_channels=50, out_channels=100, kernel_size=7, stride=2, dropout=self.config[self.DROPOUT]),
            self.__make_1d_block(in_channels=100, out_channels=250, kernel_size=7, stride=2, dropout=self.config[self.DROPOUT]),
            self.__make_1d_block(in_channels=250, out_channels=250, kernel_size=7, stride=2, dropout=self.config[self.DROPOUT]),
            self.__make_1d_block(in_channels=250, out_channels=250, kernel_size=7, stride=2, dropout=self.config[self.DROPOUT]),
            self.__make_1d_block(in_channels=250, out_channels=250, kernel_size=7, stride=1, dropout=self.config[self.DROPOUT]),
            self.__make_1d_block(in_channels=250, out_channels=250, kernel_size=7, stride=1, dropout=self.config[self.DROPOUT]),
            nn.AdaptiveMaxPool1d(output_size=1)
        )

        self.fc0 = nn.Sequential(
            self.__make_linear(in_features=250, out_features=128, dropout=self.config[self.DROPOUT]),
            self.__make_linear(in_features=128, out_features=128, dropout=self.config[self.DROPOUT]),
            self.__make_linear(in_features=128, out_features=64, dropout=self.config[self.DROPOUT])
        )

        self.feature_extractor_1 = nn.Sequential(
            self.__make_1d_block(in_channels=1, out_channels=25, kernel_size=1024, stride=256, dropout=self.config[self.DROPOUT]),
            self.__make_1d_block(in_channels=25, out_channels=50, kernel_size=7, stride=2, dropout=self.config[self.DROPOUT]),
            self.__make_1d_block(in_channels=50, out_channels=100, kernel_size=7, stride=2, dropout=self.config[self.DROPOUT]),
            self.__make_1d_block(in_channels=100, out_channels=250, kernel_size=7, stride=2, dropout=self.config[self.DROPOUT]),
            self.__make_1d_block(in_channels=250, out_channels=250, kernel_size=7, stride=2, dropout=self.config[self.DROPOUT]),
            self.__make_1d_block(in_channels=250, out_channels=250, kernel_size=7, stride=2, dropout=self.config[self.DROPOUT]),
            nn.AdaptiveMaxPool1d(output_size=1)
        )

        self.fc1 = nn.Sequential(
            self.__make_linear(in_features=250, out_features=128, dropout=self.config[self.DROPOUT]),
            self.__make_linear(in_features=128, out_features=128, dropout=self.config[self.DROPOUT]),
            self.__make_linear(in_features=128, out_features=64, dropout=self.config[self.DROPOUT])
        )

        self.feature_extractor_2 = nn.Sequential(
            self.__make_1d_block(in_channels=1, out_channels=25, kernel_size=11025, stride=2756, dropout=self.config[self.DROPOUT]),
            self.__make_1d_block(in_channels=25, out_channels=50, kernel_size=7, stride=1, dropout=self.config[self.DROPOUT]),
            self.__make_1d_block(in_channels=50, out_channels=100, kernel_size=7, stride=1, dropout=self.config[self.DROPOUT]),
            self.__make_1d_block(in_channels=100, out_channels=250, kernel_size=7, stride=1, dropout=self.config[self.DROPOUT]),
            self.__make_1d_block(in_channels=250, out_channels=250, kernel_size=7, stride=1, dropout=self.config[self.DROPOUT]),
            nn.AdaptiveMaxPool1d(output_size=1)
        )

        self.fc2 = nn.Sequential(
            self.__make_linear(in_features=250, out_features=128, dropout=self.config[self.DROPOUT]),
            self.__make_linear(in_features=128, out_features=128, dropout=self.config[self.DROPOUT]),
            self.__make_linear(in_features=128, out_features=64, dropout=self.config[self.DROPOUT])
        )

        input_size = 64 * 3

        self.fc = nn.Sequential(
            self.__make_linear(in_features=input_size, out_features=128, dropout=self.config[self.DROPOUT]),
            self.__make_linear(in_features=128, out_features=64, dropout=self.config[self.DROPOUT]),
            self.__make_linear(in_features=64, out_features=4, dropout=None, batch_normalize=False, activation=None)
        )

    def forward(self, x):

        x0 = self.feature_extractor_0(x)
        x1 = self.feature_extractor_1(x)
        x2 = self.feature_extractor_2(x)

        x0 = torch.flatten(x0, start_dim=1)
        x1 = torch.flatten(x1, start_dim=1)
        x2 = torch.flatten(x2, start_dim=1)

        x0 = self.fc0(x0)
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)

        x = torch.cat((x0, x1, x2), dim=1)

        x = self.fc(x)
        return x
