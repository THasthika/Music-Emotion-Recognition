import torchmetrics as tm

import torch
import torch.nn as nn
import torch.nn.functional as F

from pretrain_models import Wavegram_Logmel_Cnn14

from models.base import WandbBaseModel

"""
ModelCatB - A 1D convolutional Model with a Pretrained Model

BaseModel (Wavegram_Logmel_Cnn14)
FullyConnected1
FullyConnected2
FullyConnected3
Softmax

"""

class ModelCatB(WandbBaseModel):

    CMDS = [
        ('lr', float, 0.001),
        ('max_epochs', int, 100)
    ]

    def __init__(self, batch_size=32, num_workers=4, data_artifact=None, split_artifact=None, init_base=True, **config):
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            sample_rate=32000,
            chunk_duration=5,
            overlap=2.5,
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

        if init_base:
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

        ## metrics
        self.train_acc = tm.Accuracy(top_k=3)
        self.val_acc = tm.Accuracy(top_k=3)
        
        self.test_acc = tm.Accuracy(top_k=3)
        self.test_f1_class = tm.F1(num_classes=4, average='none')
        self.test_f1_global = tm.F1(num_classes=4)

        ## loss
        self.loss = F.cross_entropy

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