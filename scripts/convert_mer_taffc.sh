#!/bin/bash

MAIN_DIR=./datasets/MER_taffc/

## CONVERT MP3 to WAV

mkdir -p $MAIN_DIR/wav/Q1/
mkdir -p $MAIN_DIR/wav/Q2/
mkdir -p $MAIN_DIR/wav/Q3/
mkdir -p $MAIN_DIR/wav/Q4/

echo "Converting to wav..."

bash ./scripts/convertdir.sh $MAIN_DIR/Q1/ $MAIN_DIR/wav/Q1/ 2>&1 > /dev/null
echo "Q1 Done..."

bash ./scripts/convertdir.sh $MAIN_DIR/Q2/ $MAIN_DIR/wav/Q2/ 2>&1 > /dev/null
echo "Q2 Done..."

bash ./scripts/convertdir.sh $MAIN_DIR/Q3/ $MAIN_DIR/wav/Q3/ 2>&1 > /dev/null
echo "Q3 Done..."

bash ./scripts/convertdir.sh $MAIN_DIR/Q4/ $MAIN_DIR/wav/Q4/ 2>&1 > /dev/null
echo "Q4 Done..."

## Convert sample rates into 44100

mkdir -p $MAIN_DIR/wav4/Q1/
mkdir -p $MAIN_DIR/wav4/Q2/
mkdir -p $MAIN_DIR/wav4/Q3/
mkdir -p $MAIN_DIR/wav4/Q4/

echo "Resampling to 441000..."

bash ./scripts/wav2sr44100.sh $MAIN_DIR/wav/Q1/ $MAIN_DIR/wav4/Q1/ > /dev/null
echo "Q1 Done..."

bash ./scripts/wav2sr44100.sh $MAIN_DIR/wav/Q2/ $MAIN_DIR/wav4/Q2/ > /dev/null
echo "Q2 Done..."

bash ./scripts/wav2sr44100.sh $MAIN_DIR/wav/Q3/ $MAIN_DIR/wav4/Q3/ > /dev/null
echo "Q3 Done..."

bash ./scripts/wav2sr44100.sh $MAIN_DIR/wav/Q4/ $MAIN_DIR/wav4/Q4/ > /dev/null
echo "Q4 Done..."

# move wav4 to wav

rm -rf $MAIN_DIR/wav/
mv $MAIN_DIR/wav4/ $MAIN_DIR/wav/