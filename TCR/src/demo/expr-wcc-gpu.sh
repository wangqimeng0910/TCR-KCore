#!/bin/bash

datasets="soc-livejournal.txt"

for dataset in $datasets; do
    for count in {1..1}; do
        echo $dataset $count
        python3 ConnectedComponents.py --graph ../../data/$dataset --output /dev/null --cuda
    done
done
