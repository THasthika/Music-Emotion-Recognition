#!/bin/bash

conda create -y -n mer python=3.8

conda activate mer

conda install -y pytorch torchvision torchaudio torchtext cpuonly -c pytorch-lts

cd /home/ec2-user/SageMaker/mer_research/

bash ./install-dep.sh

conda install -c anaconda ipykernel

python -m ipykernel install --user --name=mer