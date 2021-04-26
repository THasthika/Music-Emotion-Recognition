import torchmetrics as tm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio
import torchaudio.transforms as audioT
import torchaudio.functional as audioF

from models.base import BaseModel

"""
ModelStatC - Spectrogram converted 2D Convolutional model
"""

class ModelStatC(BaseModel):

    CMDS = [
        ('lr', float, 0.001),
        ('max_epochs', int, 100),
        ('n_fft', int, 400),
        ('hidden_states', int, 64)
    ]

    def __init__(self, batch_size=32, num_workers=4, data_artifact=None, split_artifact=None, init_base=True, **config):
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            sample_rate=32000,
            duration=30,
            data_artifact=data_artifact,
            split_artifact=split_artifact,
            label_type="static")

        self.config = config
        self.lr = config['lr']

        ## build layers
        self.__build()

        ## metrics
        self.train_r2 = tm.R2Score(num_outputs=4)
        self.val_r2 = tm.R2Score(num_outputs=4)
        self.test_r2 = tm.R2Score(num_outputs=4)

        ## loss
        self.loss = F.l1_loss

    def __build(self):
        n_fft = self.config['n_fft']
        
        self.wav2spec = audioT.Spectrogram(n_fft=n_fft, window_fn=torch.hamming_window)
        
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=(n_fft//2)+1, out_channels=250, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=6, stride=2),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=6, stride=2),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=6, stride=2),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=6, stride=2),
            nn.Conv1d(in_channels=250, out_channels=64, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=6, stride=2),
            nn.AdaptiveAvgPool1d(128)
        )

        self.rnn_layer = nn.LSTM(input_size=128, hidden_size=self.config['hidden_states'], num_layers=64)

        self.predictor = nn.Sequential(
            nn.Linear(in_features=self.config['hidden_states']*64, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=4)
        )

    def forward(self, x):
        x = self.wav2spec(x)
        x = torch.squeeze(x, dim=1)
        x = self.feature_extractor(x)
        x, _ = self.rnn_layer(x)
        x = torch.flatten(x, start_dim=1)
        x = self.predictor(x)
        return x

    def predict(self, x):
        return self.forward(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_logit = self(x)
        loss = self.loss(y_logit, y)

        self.log('train/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train/r2', self.train_r2(y_logit, y), on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_logit = self(x)
        loss = self.loss(y_logit, y)
        
        self.log("val/loss", loss, prog_bar=True)
        self.log('val/r2', self.val_r2(y_logit, y))

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_logit = self(x)
        loss = self.loss(y_logit, y)

        self.log("test/loss", loss)
        self.log("test/r2", self.val_r2(y_logit, y))