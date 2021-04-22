#!/bin/bash

for f in $1/*.mp3; do
    ./scripts/stereotomono.sh $f $2
done