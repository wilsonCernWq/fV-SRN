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

# ./train_comparison.sh mechhand    640 220 229      640*220*229 \
#     mechhand-20220926010852-network.json \
#     60 16 64:64:64:64    # number of params SNR=3473665, INR=3462656

# ./train_comparison.sh chameleon     1024 1024 1080  512*512*540 \
#     chameleon-20220916174935-network.json \
#     98 16 64:64:64       # number of params SNR=15072577, INR=14992896

# ./train_comparison.sh supernova       432 432 432    432*432*432 \
#     supernova-20220916175236-network.json \
#     66 16 64:64          # number of params SNR=4609281, INR=4503040

./train_comparison.sh heatrelease_1atm     1152 320 853    1152*320*853  \
    1atmheatrelease-20220916175751-network.json \
    98 16 64:64:64:64    # number of params SNR=15076737, INR=14996992 (99.5%)

./train_comparison.sh heatrelease_10atm    1152 426 853    1152*426*853  \
    10atmheatrelease-20220916180308-network.json \
    98 16 64:64:64:64    # number of params SNR=15076737, INR=14996992 (99.5%)

./train_comparison.sh temp_1atm      1152 320 853    1152*320*853  \
    1atmtemp-20220916180933-network.json  \
    114 16 64:64:64      # number of params SNR=23718209, INR=23382528 (98.6%)

./train_comparison.sh temp_10atm     1152 426 853    1152*426*853  \
    10atmtemp-20220916181605-network.json \
    114 16 64:64:64:64   # number of params SNR=23722369, INR=23386624 (98.6%)
