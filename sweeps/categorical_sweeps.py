import wandb

from os import path
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchaudio

import torchmetrics as tm

import pytorch_lightning as pl
# from pytorch_lightning
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import random

import numpy as np
import pandas as pd

PROJECT_NAME = "mer"

# DATA_ARTIFACT = "mer-taffc:latest"
# SPLIT_ARTIFACT = "mer-taffc-train-70-val-20-test-10-seed-42:latest"

DATA_ARTIFACT = "deam:latest"
SPLIT_ARTIFACT = "deam-train-70-val-20-test-10-seed-42:latest"

class BaseModel(pl.LightningModule):

    def __init__(self, batch_size=32, num_workers=4, data_at_name=None, split_at_name=None, **config):
        """[summary]

        Args:
            batch_size (int, optional): [description]. Defaults to 32.
            num_workers (int, optional): [description]. Defaults to 4.
            data_at_name ([type], optional): [description]. Defaults to None.
            split_at_name ([type], optional): [description]. Defaults to None.
        """
        super().__init__()
        self.config = config
        self.lr = config['lr']
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data_at_name = data_at_name
        self.split_at_name = split_at_name

        ## build layers
        self.__build()

        ## metrics
        self.train_acc = tm.Accuracy(top_k=3)
        self.val_acc = tm.Accuracy(top_k=3)
        self.test_acc = tm.Accuracy(top_k=3)

        ## loss
        self.loss = F.cross_entropy

    def __build(self):

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=self.config['conv1_kernel_size'], stride=self.config['conv1_kernel_stride'])
        self.bn1 = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=64, stride=8)
        self.bn2 = nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16, stride=4)
        self.bn3 = nn.BatchNorm1d(64)

        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=self.config['adaptive_layer'])
        self.bn4 = nn.BatchNorm1d(64)

        self.fc1 = nn.Linear(in_features=self.config['adaptive_layer']*64, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=4)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.avg_pool(x)))

        x = torch.flatten(x, start_dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

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

    def prepare_data(self):
        split_at = wandb.use_artifact(self.split_at_name, type="data-split")
        split_dir = split_at.download()

        data_at = wandb.use_artifact(self.data_at_name, type="data-raw")
        data_dir = data_at.download()

        audio_dir = path.join(data_dir, "audio")
        train_meta_file = path.join(split_dir, "train.json")
        val_meta_file = path.join(split_dir, "val.json")
        test_meta_file = path.join(split_dir, "test.json")

        has_val = False
        has_test = False
        if path.exists(val_meta_file):
            has_val = True
        if path.exists(test_meta_file):
            has_test = True

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        DSClass = None
        if self.data_at_name.startswith("mer-taffc"):
            DSClass = MERTaffcDataset
        elif self.data_at_name.startswith("deam"):
            DSClass = DeamDataset

        self.train_ds = DSClass(
            audio_dir=audio_dir,
            meta_file=train_meta_file)

        if has_val:
            self.val_ds = DSClass(
                audio_dir=audio_dir,
                meta_file=val_meta_file)

        if has_test:
            self.test_ds = DSClass(
                audio_dir=audio_dir,
                meta_file=test_meta_file)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        if self.val_ds is None: return None
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        if self.test_ds is None: return None
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)

def makeTrainer(project_name, data_name, split_name, batch_size=64, tune=False):

    def train(config=None):
        if config is None:
            config = {
                'adaptive_layer': 128,
                'lr': 0.001,
                'max_epochs': 100,
                'conv1_kernel_size': 11025,
                'conv1_kernel_stride': 400
            }

        run = wandb.init(config=config, project=project_name, job_type="train")

        config = dict(run.config)
        # config = config_defaults

        print("config:", config)

        logger = WandbLogger(project=project_name, experiment=run, log_model=True)

        model = BaseModel(batch_size=batch_size, split_at_name=split_name, data_at_name=data_name, **config)

        checkpoint_callback = ModelCheckpoint(monitor='val/acc')

        trainer = pl.Trainer(
            logger=logger,
            gpus=-1,
            checkpoint_callback=True,
            progress_bar_refresh_rate=1,
            max_epochs=config['max_epochs'],
            auto_scale_batch_size=tune)

        if tune:
            trainer.tune(model)

        trainer.fit(model)

        if not model.test_ds is None:
            trainer.test(model, ckpt_path='best')
    
    return train

train = makeTrainer(PROJECT_NAME, DATA_ARTIFACT, SPLIT_ARTIFACT, batch_size=128)