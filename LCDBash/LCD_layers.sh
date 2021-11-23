#!/bin/bash

device=7
domains=3
times=2
LCDlayers=(1 2 3 4)

for domain in `seq 0 $domains`
  do
    max=$((${#LCDlayers[@]}-1))
    for l in `seq 1 1`
    do
      for i in `seq 0 ${times}`
        do
        python ../train_diversity.py \
            --target=$domain \
            --device=$device \
            --time=$i \
            --LCD_Layer=${LCDlayers[l]} \
            --CD_flag=0 \
            --CD_drop_max=0.6 \
            --CD_drop_min=0. \
            --LSR_filter_flag=0 \
            --LSR_response_flag=0 \
            --alpha=100 \
            --beta=1 \
            --Conv_or_Convs_flag=0 \
            --Attention_flag=0 \
            --mixup_flag=0 \
            --LCD_prob=10 \
            --progressive_dropout_flag=1 \
            --progressive_dropout_radio 0. 0.2 0.4 \
            --progressive_dropout_all_epoch 4 8 12 30 \
            --progressive_dropout_all_radio 0.2 \
            --learning_rate=0.001
        done
    done
  done
