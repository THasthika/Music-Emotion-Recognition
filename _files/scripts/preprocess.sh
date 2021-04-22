#!/bin/bash

PREPROCESS_DIR=./src/preprocess

script_path=$PREPROCESS_DIR/$1\.py

python $script_path
