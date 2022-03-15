#!/bin/bash

device=0
data='PACS'
network='resnet18'
baselineFlag=1
baselineProgressiveFlag=1
baselineP=(0.2 0.25 0.33 0.4)
dropoutMode=1
velocity=4
ChannelorSpatial=0
spatialBlock=1
grayFlag=1

for t in `seq 0 0`
do
  for domain in `seq 0 0`
  do
    for k in `seq 2 2`
    do
      python train_PLACE.py \
      --target $domain \
      --device $device \
      --network $network \
      --time $t \
      --batch_size 64 \
      --data $data \
      --SwapStyle_flag 1 \
      --SwapStyle_Layers 3 \
      --SwapStyle_detach_flag 0 \
      --RandAug_flag 1 \
      --m 4 \
      --n 8 \
      --stage1_flag 1 \
      --stage2_flag 1 \
      --stage2_one_stage_flag 0 \
      --stage2_stage1_LastOrBest 0 \
      --stage2_layers 1 2 3 \
      --update_parameters_method 0 \
      --random_dropout_layers_flag 1 \
      --not_before_flag 0 \
      --epochs 30 \
      --learning_rate 0.001 \
      --epochs_PLACE 30 \
      --learning_rate_style 0.001 \
      --baseline_dropout_flag $baselineFlag \
      --baseline_dropout_p ${baselineP[k]} \
      --baseline_progressive_flag ${baselineProgressiveFlag} \
      --dropout_mode ${dropoutMode} \
      --velocity ${velocity} \
      --ChannelorSpatial ${ChannelorSpatial} \
      --spatialBlock ${spatialBlock} \
      --dropout_recover_flag 1 \
      --gray_flag $grayFlag
    done
  done
done


