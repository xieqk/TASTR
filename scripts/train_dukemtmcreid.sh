#!/bin/bash

gpu='0,1'
time_now=$(date "+%Y%m%d-%H%M%S")
save_dir="./logs/dukemtmcreid"

python train_img_s1.py -g ${gpu} --save-dir ${save_dir}/${time_now}/s1

sleep 5

python train_img_s2.py -g ${gpu} --resume ${save_dir}/${time_now}/s1/s1_checkpoint_final.pth.tar --save-dir ${save_dir}/${time_now}
