#!/bin/bash

device=7
domains=3
times=3
LCD_layers=(1 2 3 4)

for domain in `seq 0 $domains`
  do
    for i in `seq 0 ${times-1}`
      do
      python train_diversity.py \
          --target=$domain \
          --device=$device \
          --time=$i \
          --LCD_Layer=1 \
          --CD_flag=0 \
          --CD_drop_max=0.8 \
          --CD_drop_min=0 \
          --LSR_filter_flag=0 \
          --LSR_response_flag=0 \
          --alpha=100 \
          --beta=1 \
          --Conv_or_Convs_flag=0
      done
  done
