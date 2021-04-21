"""
A simple implementation of a model using pytorch lightning 
"""

import os
import os.path as path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import torchaudio

import pytorch_lightning as pl
import torchmetrics as tm

from decouple import config

class MERTaffcDataSet(Dataset):
    """MER_Taffc dataset."""
    
    folders = ["Q1", "Q2", "Q3", "Q4"]
    labels = {
        "Q1": torch.tensor(0),
        "Q2": torch.tensor(1),
        "Q3": torch.tensor(2),
        "Q4": torch.tensor(3)
    }
    
    def __init__(self, root_dir, sample_rate=22050, duration=30):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.frame_count = self.sample_rate * self.duration
        
        all_files = []
        for f in self.folders:
            files = os.listdir(path.join(root_dir, f))
            files = list(map(lambda x: (path.join(root_dir, f, x), f), files))
            all_files.extend(files)
            
        self.files = all_files
        self.count = len(all_files)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        audio_file, label = self.files[idx]
        
        x, sr = torchaudio.load(audio_file)
        x = torch.mean(x, 0, True)
        out = torch.zeros(1, self.frame_count)
        
        effects = [
          ["rate", f"{self.sample_rate}"]
        ]
        
        x, sr2 = torchaudio.sox_effects.apply_effects_tensor(x, sr, effects)
        
        if self.frame_count >= x.shape[1]:
            out[:, :x.shape[1]] = x
        else:
            out[:, :] = x[:, :self.frame_count]

        # out = torch.squeeze(out)
        
        return (out, self.labels[label])

class MyCustomModel(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 48, kernel_size=11025, stride=512),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool1d(4, 2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(48, 128, 6, 3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(),
            nn.AdaptiveAvgPool1d(100)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(128*100, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout()
        )

        self.fc2 = nn.Linear(512, 4)

        # Loss:
        self.loss = F.cross_entropy

        # Metrics:
        self.train_acc = tm.Accuracy()
        self.valid_acc = tm.Accuracy()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        z = self.forward(x)
        loss = F.cross_entropy(z, y)
        z_hat = F.softmax(z, dim=1)
        self.train_acc(z_hat, y)

        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
DATA_PATH = config("DATA_PATH")
GPUS = int(config("GPUS", default=0))
BATCH_SIZE = int(config("BATCH_SIZE", default=32))
NUM_WORKERS = int(config("NUM_WORKERS", default=4))

dataset = MERTaffcDataSet(DATA_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

model = MyCustomModel()
trainer = pl.Trainer(gpus=GPUS)

trainer.fit(model, dataloader)