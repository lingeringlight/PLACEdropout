#!/bin/bash

device=5
domains=3
times=4
LCDlayers=(1 2 3 4)

for domain in `seq 0 $domains`
  do
    max=$((${#LCDlayers[@]}-1))
    for l in `seq 1 1`
    do
      for i in `seq 0 ${times}`
        do
        python ../train_Mixup.py \
            --target=$domain \
            --device=$device \
            --time=$i \
            --MixStyle_flag=1 \
            --learning_rate=0.001 \
            --data='PACS'
        done
    done
  done
