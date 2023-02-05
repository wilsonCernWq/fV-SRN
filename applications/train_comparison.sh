#!/bin/bash

# data=chameleon
# dimx=1024
# dimy=1024
# dimz=1080
# config=chameleon-20220916174935-network.json
# latent_res=98
# latent_fts=16
# layers=64:64:64

data=$1
dimx=$2
dimy=$3
dimz=$4
samples=$5
config=$6
latent_res=$7
latent_fts=$8
layers=$9

config=/home/qadwu/Work/instant-vnr-cuda/data/threshold-compress/compress-200000/${config}

# HYBRID TRAINING
python volnet/train_volnet_lite.py \
   $PWD/config-files/instant-vnr/$data.json \
   --train:mode world \
   --train:samples ${samples} \
   --train:sampler_importance 0.01 \
   --train:batchsize 64*64*128 \
   --rebuild_dataset 0 \
   --val:copy_and_split \
   --outputmode density:direct \
   --lossmode density \
   --layers ${layers} \
   --activation SnakeAlt:1 \
   --fouriercount 30 \
   --fourierstd -1 \
   --volumetric_features_resolution ${latent_res} \
   --volumetric_features_channels   ${latent_fts} \
   -l1 0.8 -l2 0.2 \
   -lr 0.005 \
   --lr_step 5 --lr_gamma 0.8 \
   -i 200 \
   --logdir   volnet/results/${data}_hybrid/srn/log   \
   --modeldir volnet/results/${data}_hybrid/srn/model \
   --hdf5dir  volnet/results/${data}_hybrid/srn/hdf5  \
   --save_frequency 20 \
   --dims $dimx $dimy $dimz ${@:10}

# INR TRAINING (do not rebuild dataset)
python volnet/train_volnet_lite.py \
   $PWD/config-files/instant-vnr/$data.json \
   --train:mode world \
   --train:samples ${samples} \
   --train:sampler_importance 0.01 \
   --train:batchsize 64*64*128 \
   --rebuild_dataset 0 \
   --val:copy_and_split \
   --lossmode density \
   -l1 0.8 -l2 0.2 \
   -lr 0.005 \
   --lr_step 5 --lr_gamma 0.8 \
   -i 200 \
   --logdir   volnet/results/${data}_hybrid/inr/log   \
   --modeldir volnet/results/${data}_hybrid/inr/model \
   --hdf5dir  volnet/results/${data}_hybrid/inr/hdf5  \
   --save_frequency 20 \
   --dims $dimx $dimy $dimz \
   --inr $config ${@:10}
