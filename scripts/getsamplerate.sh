#!/bin/bash

ret=$(ffprobe -show_streams $1 2> /dev/null | grep "sample_rate" | sed "s/sample_rate=//")

echo $ret