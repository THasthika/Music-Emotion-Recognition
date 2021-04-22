#!/bin/bash

pval=0

for f in $1/*.wav; do
  v=$(ffprobe -show_streams $f 2> /dev/null | grep sample_rate | sed "s/sample_rate=//")
  if [[ $v -ne $pval ]]
  then
    pval=$v
    echo $pval
  fi
done