from models import BaseCatModel
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchmetrics as tm

class A1DConvLSTMCat_V1(BaseCatModel):

    HIDDEN_SIZE = "hidden_size"
    NUM_LAYERS = "num_layers"

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

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=250, kernel_size=1024, stride=256),
            nn.BatchNorm1d(250),
            nn.Dropout(p=self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1),
            nn.BatchNorm1d(250),
            nn.Dropout(p=self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1),
            nn.BatchNorm1d(250),
            nn.Dropout(p=self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1),
            nn.BatchNorm1d(250),
            nn.Dropout(p=self.config[self.DROPOUT]),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=250,
            hidden_size=self.config[self.HIDDEN_SIZE],
            num_layers=self.config[self.NUM_LAYERS]
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.config[self.HIDDEN_SIZE], out_features=512),
            nn.Dropout(p=self.config[self.DROPOUT]),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=4)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.permute((0, 2, 1))
        (out, _) = self.lstm(x)
        x = out[:, -1, :]
        x = self.fc(x)
        return x
        