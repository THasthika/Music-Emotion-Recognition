import torch
import torch.nn as nn

from models import BaseCatModel


class A1DConvCat_V1(BaseCatModel):
    ADAPTIVE_LAYER_UNITS = "adaptive_layer_units"

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
            nn.Dropout(self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.BatchNorm1d(250),
            nn.Dropout(self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.BatchNorm1d(250),
            nn.Dropout(self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.BatchNorm1d(250),
            nn.Dropout(self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.BatchNorm1d(250),
            nn.Dropout(self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.BatchNorm1d(250),
            nn.Dropout(self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.BatchNorm1d(250),
            nn.Dropout(self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.BatchNorm1d(250),
            nn.Dropout(self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(output_size=self.config[self.ADAPTIVE_LAYER_UNITS]),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.config[self.ADAPTIVE_LAYER_UNITS] * 250, out_features=512),
            nn.Dropout(self.config[self.DROPOUT]),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=4)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
