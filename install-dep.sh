#!/bin/bash

# wandb
pip3 install -qqq wandb

# pytorch
pip3 install -qqq torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 torchtext==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# pytorch_lightning
pip3 install -qqq pytorch-lightning

# torchmetrics
pip3 install -qqq torchmetrics

pip3 install -qqq torchaudio

# torchlibrosa
pip3 install -qqq torchlibrosa

# dotenv
pip3 install -qqq python-dotenv

# torchinfo
pip3 install -qqq torchinfo

# nnaudio
pip3 install -qqq nnAudio

# essentia
pip3 install -qqq essentia

#pylrc
pip3 install -qqq git+https://github.com/THasthika/pylrc.git@timestamp-search

# trainsformers
pip3 install -qqq transformers

# coolname
pip3 install -qqq coolname

# librosa
pip3 install -qqq librosa