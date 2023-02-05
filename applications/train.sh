#!/bin/bash

# ONLY NETWORK TRAINING
# python volnet/train_volnet_lite.py \
#    $PWD/config-files/instant-vnr/$1.json \
#    --train:mode world \
#    --train:samples 256**3 \
#    --train:sampler_importance 0.01 \
#    --train:batchsize 64*64*128 \
#    --rebuild_dataset 51 \
#    --val:copy_and_split \
#    --outputmode density:direct \
#    --lossmode density \
#    --layers 256:256:256:256:256:256:256:256:256 \
#    --activation ResidualSine \
#    --fouriercount 0 \
#    -l1 1 \
#    -lr 0.00005 \
#    --lr_step 100 \
#    -i 200 \
#    --logdir volnet/results/$1_256/log \
#    --modeldir volnet/results/$1_256/model \
#    --hdf5dir volnet/results/$1_256/hdf5 \
#    --save_frequency 20

# HYBRID TRAINING
python volnet/train_volnet_lite.py \
   $PWD/config-files/instant-vnr/$1.json \
   --train:mode world \
   --train:samples 256**3 \
   --train:sampler_importance 0.01 \
   --train:batchsize 64*64*128 \
   --rebuild_dataset 101 \
   --val:copy_and_split \
   --outputmode density:direct \
   --lossmode density \
   --layers 64:64:64:64 \
   --activation SnakeAlt:1 \
   --fouriercount 30 \
   --fourierstd -1 \
   --volumetric_features_resolution 96 \
   --volumetric_features_channels 32 \
   -l1 1 \
   -lr 0.01 \
   --lr_step 10 \
   -i 200 \
   --logdir   volnet/results/test/srn/log   \
   --modeldir volnet/results/test/srn/model \
   --hdf5dir  volnet/results/test/srn/hdf5  \
   --save_frequency 20 \
   --dims $2 $3 $4

# # INR TRAINING (do not rebuild dataset)
# python volnet/train_volnet_lite.py \
#    $PWD/config-files/instant-vnr/$1.json \
#    --train:mode world \
#    --train:samples 256**3 \
#    --train:sampler_importance 0.01 \
#    --train:batchsize 64*64*128 \
#    --rebuild_dataset 0 \
#    --val:copy_and_split \
#    --lossmode density \
#    -l1 1 \
#    -lr 0.01 \
#    --lr_step 10 \
#    -i 10 \
#    --logdir   volnet/results/test/inr/log   \
#    --modeldir volnet/results/test/inr/model \
#    --hdf5dir  volnet/results/test/inr/hdf5  \
#    --save_frequency 20 \
#    --dims $2 $3 $4 \
#    --inr $5
