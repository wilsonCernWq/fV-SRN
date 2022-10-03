#!/bin/bash

./train.sh chameleon         1024 1024 1080;
./train.sh heatrelease_1atm  1152 320  853 ;
./train.sh heatrelease_10atm 1152 426  853 ;
./train.sh mechhand          640  220  229 ;
./train.sh supernova         432  432  432 ;
./train.sh temp_1atm         1152 320  853 ;
./train.sh temp_10atm        1152 426  853 ;