#!/bin/bash

bname=$(basename $1)
fname=$2/${bname%.mp3}.wav
ffmpeg -i $1 $fname