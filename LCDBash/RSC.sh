#!/bin/bash

device=0
data='VLCS'
network='resnet18'
grayFlag=1

for t in `seq 0 0`
do
for domain in `seq 1 3`
  do
              python ../train.py \
                  --target $domain \
                  --device $device \
                  --network $network \
                  --batch_size 128 \
                  --data $data \
                  --epochs 30 \
                  --learning_rate 0.004 \
                  --gray_flag $grayFlag
  done
done
