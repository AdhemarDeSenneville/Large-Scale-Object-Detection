#!/bin/bash

python /home/adhemar/Bureau/METHAN/code/githubs/Large-Selective-Kernel-Network/tools/train.py\
   /home/adhemar/Bureau/METHAN/code/config/experiments/lsk/lsk_s_fpn_1x_dota_le90_finetuned_terra_iter_1.py\
      --work-dir /home/adhemar/Bureau/METHAN/code/logs/lsk_train_france_iteration_1_F\
      --resume-from /home/adhemar/Bureau/METHAN/code/logs/lsk_train_france_iteration_1_F/epoch_35.pth\