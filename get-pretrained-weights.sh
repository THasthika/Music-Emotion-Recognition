#!/bin/bash

PRETRAIN_WEIGHTS_DIR="./pretrain_weights/"

weights=(
    "Wavegram_Logmel_Cnn14_mAP=0.439.pth"
    "Cnn14_mAP=0.431.pth"
)

if [[ ! -f "$PRETRAIN_WEIGHTS_DIR" ]]; then
    mkdir -p "$PRETRAIN_WEIGHTS_DIR"
fi

for w in ${weights[@]}; do
    if [[ -f "$PRETRAIN_WEIGHTS_DIR/$w" ]]; then
        continue
    fi
    l="https://zenodo.org/record/3987831/files/$w?download=1"
    wget -O "$PRETRAIN_WEIGHTS_DIR/$w" $l
done



# CHECKPOINT_PATH="Cnn14_mAP=0.431.pth"
# wget -O $CHECKPOINT_PATH https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1