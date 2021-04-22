#!/bin/bash

# pytorch
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1  -f https://download.pytorch.org/whl/torch_stable.html

# wandb
pip3 install wandb

# pytorch_lightning
pip3 install pytorch-lightning

# torchmetrics
pip3 install git+https://github.com/PytorchLightning/metrics.git@master