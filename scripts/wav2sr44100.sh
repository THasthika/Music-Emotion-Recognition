#!/bin/bash

function change_sample_rate() {
  for i in {1.0,0.99,0.98,0.97,0.96,0.95,0.94,0.93,0.92,0.91,0.9,0.89,0.88,0.87,0.86}
  do
    a=$(sox -v $i $1 -r 44100 $2 2>&1)
    if [ -z "$a" ]
    then
      break
    fi
  done
}

for f in $1/*.wav; do
  bname=$(basename $f)
  fname=$2/$bname
  sr=$(bash ./scripts/getsamplerate.sh $f)
  if [[ $sr -eq 44100 ]]
  then
    cp $f $fname
  else
    change_sample_rate $f $fname
  fi
  echo "$bname done..."
done