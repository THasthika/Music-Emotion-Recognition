import torchmetrics as tm

import torch
import torch.nn as nn
import torch.nn.functional as F

from pretrain_models import Wavegram_Logmel_Cnn14

from models.base import BaseModel

"""
ModelStatB - A 1D convolutional Model with a Pretrained Model

BaseModel (Wavegram_Logmel_Cnn14)
FullyConnected1
FullyConnected2
FullyConnected3

"""

class ModelStatB(BaseModel):

    CMDS = [
        ('lr', float, 0.001),
        ('max_epochs', int, 100)
    ]

    def __init__(self, batch_size=32, num_workers=4, data_artifact=None, split_artifact=None, **config):
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            sample_rate=32000,
            duration=30,
            data_artifact=data_artifact,
            split_artifact=split_artifact,
            label_type="categorical")
        
        base_config = dict(
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=527
        )

        self.base = Wavegram_Logmel_Cnn14(**base_config)

        ld = torch.load("./pretrain_weights/Wavegram_Logmel_Cnn14_mAP=0.439.pth")
        self.base.load_state_dict(ld["model"])
        self.base.train()

        self.config = config
        self.lr = config['lr']

        ## freeze layers
        for x in self.base.parameters():
            x.requires_grad = False

        unfreeze_layers = [
                   'fc1',
                #    'conv_block6',
                #    'pre_block4',
                #    'conv_block5',
                #    'pre_block3'
                   ]

        for n, m in self.base.named_children():
            if n in unfreeze_layers:
                for l in m.parameters():
                    l.requires_grad = True

        ## build layers
        self.__build()

        # ## metrics
        # self.train_acc = tm.Accuracy(top_k=3)
        # self.val_acc = tm.Accuracy(top_k=3)
        # self.test_acc = tm.Accuracy(top_k=3)

        ## loss
        self.loss = F.l1_loss

    def __build(self):

        self.fc1 = nn.Linear(in_features=2048, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=4)

    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        x = self.base(x)['embedding']
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
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

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_logit = self(x)
        loss = self.loss(y_logit, y)
        
        self.log("val/loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_logit = self(x)
        loss = self.loss(y_logit, y)

        self.log("test/loss", loss)