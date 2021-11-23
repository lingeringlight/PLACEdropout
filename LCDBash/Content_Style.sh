#!/bin/bash

device=2
data='OfficeHome'
network='resnet18'
cindex=0
poolMethod=0
channelRadios=(0.5 0.7 0.9)
wClass=(1 3 5)
wDomain=(0.1 0.4 0.7)
methodIndex=1
contentFlag=0
contentProgressiveFlag=0
contentP=(0.22 0.33 0.44)
styleFlag=0
styleProgressiveFlag=0
styleP=(0.22 0.33 0.44)
baselineFlag=1
baselineProgressiveFlag=1
baselineP=(0.2 0.33 0.4)
dropoutMode=1
velocity=4
ChannelorSpatial=0
spatialBlock=1
corrDropoutFlag=0
grayFlag=1

for t in `seq 2 2`
do
for domain in `seq 3 3`
do
  for m in `seq 0 0`
  do
  for i in `seq 0 0`
    do
      for j in `seq 0 0`
        do
          for k in `seq 0 2`
          do
            python ../train_Content_Style.py \
                --target $domain \
                --device $device \
                --network $network \
                --time $t \
                --batch_size 64 \
		--data $data \
                --MixStyle_flag 0 \
                --MixStyle_Layers 3 \
                --MixStyle_detach_flag 0 \
                --MixStyle_mix_flag 0 \
                --RandAug_flag 1 \
                --m 4 \
                --n 8 \
                --label_smooth 0 \
                --stage1_flag 0 \
                --stage2_flag 1 \
                --stage2_one_stage_flag 0 \
                --stage2_stage1_LastOrBest 0 \
                --stage2_progressive_flag 0 \
                --stage2_reverse_flag 0 \
                --stage2_layers 3 4 \
		--adjust_single_layer 0 \
		--dropout_epoch_stop 0 \
		--update_parameters_method 0 \
		--random_dropout_layers_flag 1 \
		--not_before_flag 0 \
                --epochs 30 \
                --learning_rate 0.001 \
                --epochs_layer 15 \
                --epochs_style 30 \
                --learning_rate_style 0.001 \
                --method_index ${methodIndex} \
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
  done
done
done


