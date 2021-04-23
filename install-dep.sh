#!/bin/bash

# pytorch
pip3 install -qqq torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1  -f https://download.pytorch.org/whl/torch_stable.html

# wandb
pip3 install -qqq wandb

# pytorch_lightning
pip3 install -qqq pytorch-lightning

# torchmetrics
pip3 install -qqq torchmetrics