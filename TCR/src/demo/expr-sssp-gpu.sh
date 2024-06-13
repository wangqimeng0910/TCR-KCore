#!/bin/bash

datasets="ego-facebook.edges wiki-Vote.txt soc-livejournal.txt"

for dataset in $datasets; do
    for count in {1..1}; do
        echo $dataset $count
        python3 ShortestPaths.py --graph ../../data/$dataset --output /dev/null --cuda --source 0
    done
done
