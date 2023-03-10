#!/bin/bash

# ./train.sh chameleon         1024 1024 1080;
# ./train.sh heatrelease_1atm  1152 320  853 ;
# ./train.sh heatrelease_10atm 1152 426  853 ;
# ./train.sh mechhand          640  220  229 ;
# ./train.sh supernova         432  432  432 ;
# ./train.sh temp_1atm         1152 320  853 ;
# ./train.sh temp_10atm        1152 426  853 ;

# data=chameleon
# dimx=1024
# dimy=1024
# dimz=1080
# config=chameleon-20220916174935-network.json
# latent_res=98
# latent_fts=16
# layers=64:64:64

# CMD:  ['/home/qadwu/Software/miniconda3/envs/proj-fvsrn/bin/python', 'volnet/train_volnet.py', 'config-files/RichtmyerMeshkov-t60-v1-dvr.json', '--train:mode', 'world', '--train:samples', '256**3', '--train:sampler_importance', '0.01', '--rebuild_dataset', '51', '--val:copy_and_split', '--outputmode', 'density:direct', '--lossmode', 'density', '-l1', '1', '--lr_step', '100', '-i', '200', '--logdir', 'volnet/results/eval_CompressionTeaser/log', '--modeldir', 'volnet/results/eval_CompressionTeaser/model', '--hdf5dir', 'volnet/results/eval_CompressionTeaser/hdf5', '--save_frequency', '20', '--layers', '128:128:128:128:128:128:128:128:128', '--train:batchsize', '64*64*32', '--activation', 'ResidualSine', '--fouriercount', '0', '-lr', '0.00005', '--name', 'rm60-OnlyNetwork'] volnet/results/eval_CompressionTeaser/hdf5/rm60-OnlyNetwork.hdf5

# CMD:  ['/home/qadwu/Software/miniconda3/envs/proj-fvsrn/bin/python', 'volnet/train_volnet.py', 'config-files/RichtmyerMeshkov-t60-v1-dvr.json', '--train:mode', 'world', '--train:samples', '256**3', '--train:sampler_importance', '0.01', '--rebuild_dataset', '51', '--val:copy_and_split', '--outputmode', 'density:direct', '--lossmode', 'density', '-l1', '1', '--lr_step', '100', '-i', '200', '--logdir', 'volnet/results/eval_CompressionTeaser/log', '--modeldir', 'volnet/results/eval_CompressionTeaser/model', '--hdf5dir', 'volnet/results/eval_CompressionTeaser/hdf5', '--save_frequency', '20', '--layers', '32:32:32', '--train:batchsize', '64*64*128', '--activation', 'SnakeAlt:1', '--fouriercount', '14', '--fourierstd', '-1', '--volumetric_features_channels', '16', '--volumetric_features_resolution', '32', '-lr', '0.01', '--name', 'rm60-Hybrid'] volnet/results/eval_CompressionTeaser/hdf5/rm60-Hybrid.hdf5

# ./train_small_samples.sh mechhand      640 220 229      \
#     mechhand-20220926010852-network.json \
#     60 16 64:64:64:64    51   # number of params SNR=3473665, INR=3462656

# ./train_small_samples.sh chameleon     1024 1024 1080   \
#     chameleon-20220916174935-network.json \
#     98 16 64:64:64       51   # number of params SNR=15072577, INR=14992896

# ./train_small_samples.sh supernova     432 432 432      \
#     supernova-20220916175236-network.json \
#     66 16 64:64          51   # number of params SNR=4609281, INR=4503040

# ./train_small_samples.sh heatrelease_1atm     1152 320 853    \
#     1atmheatrelease-20220916175751-network.json \
#     98 16 64:64:64:64    51 # number of params SNR=15076737, INR=14996992 (99.5%)

# ./train_small_samples.sh heatrelease_10atm    1152 426 853    \
#     10atmheatrelease-20220916180308-network.json \
#     98 16 64:64:64:64    51 # number of params SNR=15076737, INR=14996992 (99.5%)

# ./train_small_samples.sh temp_1atm      1152 320 853   \
#     1atmtemp-20220916180933-network.json  \
#     114 16 64:64:64      51 # number of params SNR=23718209, INR=23382528 (98.6%)

# ./train_small_samples.sh temp_10atm     1152 426 853    \
#     10atmtemp-20220916181605-network.json \
#     114 16 64:64:64:64   51 # number of params SNR=23722369, INR=23386624 (98.6%)

# # fouriercount === 14 !!!
# ./train_small_samples.sh rm_t60     256 256 256   \
#     ../models/rm_t60-network.json \
#     32 16   32:32:32   51     # number of params SNR=527969, INR=527360 (98.6%)

# # fouriercount === 14 !!!
# ./train_small_samples.sh vmhead     256 256 256   \
#     ../models/vmhead-network.json \
#     32 16   32:32:32   51       # number of params SNR=23722369, INR=23386624 (98.6%)

./train_small_samples.sh richtmyer_meshkov     2048 2048 1920   \
    richtmyer_meshkov-20220916183752-network.json \
    120 16   64:64:64:64:64       51 # --dryrun   # number of params SNR=27669825, INR=27586048

./train_small_samples.sh pigheart     2048 2048 1920   \
    pigheart-20220916185946-network.json \
    60 16   64:64:64             51  # --dryrun   # number of params SNR=3469505, INR=3459584

