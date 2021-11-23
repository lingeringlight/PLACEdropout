#!/bin/bash

device=4
domains=3
times=0
LCDlayers=(1 2 3 4)

for i in `seq 0 ${times}`
  do
    max=$((${#LCDlayers[@]}-1))
    for l in `seq 0 0`
    do
      for domain in `seq 0 $domains`
        do
        python ../train_diversity.py \
            --target=$domain \
            --device=$device \
            --time=$i \
            --LCD_Layer 3 \
            --CD_flag=0 \
            --CD_drop_max=0.6 \
            --CD_drop_min=0. \
            --LSR_filter_flag=0 \
            --LSR_response_flag=0 \
            --LSR_response_group=64 \
            --alpha=100 \
            --beta=1 \
            --Conv_or_Convs_flag=0 \
            --Attention_flag=0 \
            --mixup_flag=0 \
            --LCD_prob=10 \
            --progressive_dropout_flag=4 \
            --progressive_dropout_radio 0. 0.11 0.22 0.22 0.22 0.22 \
            --progressive_dropout_all_PCD_flag 1 \
            --progressive_dropout_all_epoch 0 10 30 30 \
            --progressive_dropout_all_radio=0.2 \
            --progressive_dropout_linear_epoch=20 \
            --progressive_dropout_linear_epoch_radio=0.015 \
            --progressive_dropout_metric_mode=0 \
            --progressive_dropout_metric_pmax=0.33 \
            --progressive_dropout_metric_sample_flag=0 \
            --progressive_dropout_recover_flag=1 \
            --group_CD_flag=0 \
            --group_channel_batch_average_flag=0 \
            --drop_group_num=3 \
            --MixStyle_flag 1 \
            --train_diversity_flag 0 \
            --test_diversity_flag 1 \
            --learning_rate=0.001 \
            --data='PACS' \
            --RandAug_flag=1 \
            --m=4 \
            --n=8 \
            --epochs=30 \
            --batch_size=64
        done
    done
  done
