#!/bin/bash

run_f=$1
shift
for r in $@
do
    python $run_f $r
done