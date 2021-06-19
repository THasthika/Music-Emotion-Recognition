#!/bin/bash

conda create -y -n mer python=3.8

conda install -y -n mer pytorch torchvision torchaudio torchtext cpuonly -c pytorch-lts

conda install -y -n mer -c anaconda ipykernel

python -m ipykernel install --user --name=mer

cp /home/ec2-user/efs/config/sagemaker-default.yaml /home/ec2-user/SageMaker/mer_research/default.yaml

cp /home/ec2-user/efs/config/mer-kernel.json /home/ec2-user/.local/share/jupyter/kernels/mer/kernel.json