#!/bin/bash

datasets="soc-sinaweibo.txt"

for dataset in $datasets; do
    for count in {1..1}; do
        echo $dataset $count
        python3 PageRankGraph.py --graph ../../data/$dataset --output /dev/null --cuda
    done
done
