import torchmetrics as tm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio
import torchaudio.transforms as audioT
import torchaudio.functional as audioF

from models.base import WandbBaseModel

"""
ModelCatC - Spectrogram converted 2D Convolutional model
"""

class ModelCatC(WandbBaseModel):

    CMDS = [
        ('lr', float, 0.001),
        ('max_epochs', int, 100),
        ('n_fft', int, 400),
        ('hidden_states', int, 64),
        ('adaptive_layer', int, 128)
    ]

    def __init__(self, batch_size=32, num_workers=4, data_artifact=None, split_artifact=None, init_base=True, **config):
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            sample_rate=22050,
            chunk_duration=5,
            overlap=2.5,
            data_artifact=data_artifact,
            split_artifact=split_artifact,
            label_type="categorical")

        self.config = config
        self.lr = config['lr']

        ## build layers
        self.__build()

        ## metrics
        self.train_acc = tm.Accuracy(top_k=3)
        self.val_acc = tm.Accuracy(top_k=3)

        self.test_acc = tm.Accuracy(top_k=3)
        self.test_f1_class = tm.F1(num_classes=4, average='none')
        self.test_f1_global = tm.F1(num_classes=4)

        ## loss
        self.loss = F.cross_entropy

    def __build(self):
        n_fft = self.config['n_fft']
        
        self.wav2spec = audioT.Spectrogram(n_fft=n_fft, window_fn=torch.hamming_window)
        
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=(n_fft//2)+1, out_channels=250, kernel_size=7, stride=1),
            nn.BatchNorm1d(num_features=250),
            nn.ReLU(),
            nn.Dropout(),
            nn.AvgPool1d(kernel_size=6, stride=2),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1),
            nn.BatchNorm1d(num_features=250),
            nn.ReLU(),
            nn.Dropout(),
            nn.AvgPool1d(kernel_size=6, stride=2),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1),
            nn.BatchNorm1d(num_features=250),
            nn.ReLU(),
            nn.Dropout(),
            nn.AvgPool1d(kernel_size=6, stride=2),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1),
            nn.BatchNorm1d(num_features=250),
            nn.ReLU(),
            nn.Dropout(),
            nn.AvgPool1d(kernel_size=6, stride=2),
            nn.Conv1d(in_channels=250, out_channels=64, kernel_size=7, stride=1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Dropout(),
            nn.AvgPool1d(kernel_size=6, stride=2),
            nn.AdaptiveAvgPool1d(self.config['adaptive_layer'])
        )

        self.rnn_layer = nn.LSTM(input_size=self.config['adaptive_layer'], hidden_size=self.config['hidden_states'], num_layers=64)

        self.predictor = nn.Sequential(
            nn.Linear(in_features=self.config['hidden_states']*64, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=4)
        )

    def forward(self, x):
        self.rnn_layer()
        x = self.wav2spec(x)
        x = torch.squeeze(x, dim=1)
        x = self.feature_extractor(x)
        x, _ = self.rnn_layer(x)
        x = torch.flatten(x, start_dim=1)
        x = self.predictor(x)
        return x

    def predict(self, x):
        x = self.forward(x)
        return F.softmax(x, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
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

# class ConvBlock(nn.Module):

