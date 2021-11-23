#!/bin/bash

device=1
domains=3
times=0
LCDlayers=(1 2 3 4)

for i in `seq 0 ${times}`
  do
    max=$((${#LCDlayers[@]}-1))
    for l in `seq 1 1`
    do
      for domain in `seq 0 $domains`
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
            --progressive_dropout_flag=0 \
            --progressive_dropout_radio 0. 0.11 0.22 0.33 0.33 0.33 \
            --progressive_dropout_all_epoch 4 8 12 30 \
            --progressive_dropout_all_radio 0.2 \
            --progressive_dropout_recover_flag 1 \
            --MixStyle_flag 0 \
            --train_diversity_flag 0 \
            --test_diversity_flag 1 \
            --learning_rate=0.001 \
            --data="OfficeHome" \
            --epochs=30
        done
    done
  done
