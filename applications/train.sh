#!/bin/bash

# ONLY NETWORK TRAINING
python volnet/train_volnet.py \
   $PWD/config-files/instant-vnr/$1.json \
   --train:mode world \
   --train:samples 256**3 \
   --train:sampler_importance 0.01 \
   --train:batchsize 64*64*128 \
   --rebuild_dataset 51 \
   --val:copy_and_split \
   --outputmode density:direct \
   --lossmode density \
   --layers 256:256:256:256:256:256:256:256:256 \
   --activation ResidualSine \
   --fouriercount 0 \
   -l1 1 \
   -lr 0.00005 \
   --lr_step 100 \
   -i 200 \
   --logdir volnet/results/$1_256/log \
   --modeldir volnet/results/$1_256/model \
   --hdf5dir volnet/results/$1_256/hdf5 \
   --save_frequency 20


# HYBRID TRAINING
# python volnet/train_volnet.py \
#    /home/qadwu/Work/fV-SRN/applications/config-files/instant-vnr/$1.json \
#    --train:mode world \
#    --train:samples 256**3 \
#    --train:sampler_importance 0.01 \
#    --train:batchsize 64*64*128 \
#    --rebuild_dataset 51 \
#    --val:copy_and_split \
#    --outputmode density:direct \
#    --lossmode density \
#    --layers 128:128:128:128:128:128:128:128:128 \
#    --activation SnakeAlt:1 \
#    --fouriercount 14 \
#    --fourierstd -1 \
#    --volumetric_features_resolution 32 \
#    --volumetric_features_channels 16 \
#    -l1 1 \
#    -lr 0.01 \
#    --lr_step 100 \
#    -i 200 \
#    --logdir volnet/results/$1/log \
#    --modeldir volnet/results/$1/model \
#    --hdf5dir volnet/results/$1/hdf5 \
#    --save_frequency 20
