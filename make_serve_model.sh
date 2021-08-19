#!/bin/bash

# torch-model-archiver --model-name testmodel --serialized-file ./testmodel.pt --version 0.0.1 --handler ./serving/custom_handler.py

# curl -X POST localhost:8080/predictions/testmodel/0.0.1 -i -F 'fileX=@./test.txt' -F 'fileY=@./test.txt'

# mv testmodel.mar model_store

# torchserve --stop