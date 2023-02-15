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
config=$5
latent_res=$6
latent_fts=$7
layers=$8
rebuild=$9

samples=256**3
config=/home/qadwu/Work/instant-vnr-cuda/data/threshold-compress/compress-200000/${config}
outdir=/mnt/scratch/ssd/qadwu/fvsrn/run03-rebuild

# HYBRID TRAINING
python volnet/train_volnet_lite.py \
   $PWD/config-files/instant-vnr/$data.json \
   --train:mode world \
   --train:samples ${samples} \
   --train:sampler_importance 0.01 \
   --train:batchsize 64*64*128 \
   --rebuild_dataset ${rebuild} \
   --val:copy_and_split \
   --outputmode density:direct \
   --lossmode density \
   --layers ${layers} \
   --activation SnakeAlt:1 \
   --fouriercount 30 \
   --fourierstd -1 \
   --volumetric_features_resolution ${latent_res} \
   --volumetric_features_channels   ${latent_fts} \
   -l1 1 \
   -lr 0.01 \
   --lr_step 120 \
   -i 200 \
   --logdir   ${outdir}/${data}_hybrid/srn/log   \
   --modeldir ${outdir}/${data}_hybrid/srn/model \
   --hdf5dir  ${outdir}/${data}_hybrid/srn/hdf5  \
   --save_frequency 20 \
   --dims $dimx $dimy $dimz ${@:10}

# INR TRAINING (do not rebuild dataset)
python volnet/train_volnet_lite.py \
   $PWD/config-files/instant-vnr/$data.json \
   --train:mode world \
   --train:samples ${samples} \
   --train:sampler_importance 0.01 \
   --train:batchsize 64*64*128 \
   --rebuild_dataset ${rebuild} \
   --val:copy_and_split \
   --lossmode density \
   -l1 1 \
   -lr 0.005 \
   --lr_step 10 --lr_gamma 0.8 \
   -i 200 \
   --logdir   ${outdir}/${data}_hybrid/inr/log   \
   --modeldir ${outdir}/${data}_hybrid/inr/model \
   --hdf5dir  ${outdir}/${data}_hybrid/inr/hdf5  \
   --save_frequency 20 \
   --dims $dimx $dimy $dimz \
   --inr $config ${@:10}
