from models.base import BaseStatModel
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchmetrics as tm
from utils.activation import CustomELU

class A2DConvStat_V1(BaseStatModel):

    ADAPTIVE_LAYER_UNITS_0 = "adaptive_layer_units_0"
    ADAPTIVE_LAYER_UNITS_1 = "adaptive_layer_units_1"

    STD_ACTIVATION = "std_activation"

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

        self.feature_1d_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=500, kernel_size=1024, stride=256),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(num_features=500),
            nn.ReLU(),
        )

        self.feature_2d_extractor = nn.Sequential(

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

        stdActivation = None
        if self.config[self.STD_ACTIVATION] == "custom":
            stdActivation = CustomELU(alpha=1.0)
        elif self.config[self.STD_ACTIVATION] == "relu":
            stdActivation = nn.ReLU()
        elif self.config[self.STD_ACTIVATION] == "softplus":
            stdActivation = nn.Softplus()

        if stdActivation is None:
            raise Exception("Activation Type Unknown!")

        self.fc_std = nn.Sequential(
            nn.Linear(in_features=128, out_features=2),
            stdActivation
        )

    def forward(self, x):
        
        x = self.feature_1d_extractor(x)
        x = torch.unsqueeze(x, dim=1)
        x = self.feature_2d_extractor(x)
        x = torch.flatten(x, start_dim=1)

        x = self.fc0(x)

        x_mean = self.fc_mean(x)
        x_std = self.fc_std(x)
        x = torch.cat((x_mean, x_std), dim=1)
        return x
    