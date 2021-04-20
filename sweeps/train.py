#!/bin/bash

import argparse

import categorical_sweeps as model

parser = argparse.ArgumentParser()
parser.add_argument('--adaptive_layer', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--conv1_kernel_size', type=int, default=11025)
parser.add_argument('--conv1_kernel_stride', type=int, default=400)

args = parser.parse_args()
options = vars(args)

model.train(config=options)
