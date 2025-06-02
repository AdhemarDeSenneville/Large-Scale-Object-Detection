#!/bin/bash

python /home/adhemar/Bureau/METHAN/code/pip_france/inference.py \
  --dataset_path="/home/adhemar/Bureau/datasets/France/tilling_france_1000p_820h" \
  --model_config="/home/adhemar/Bureau/METHAN/code/logs/train_terra_fcos_iteration_1_2/rotated_fcos_r50_fpn_1x_dota_le90.py" \
  --model_checkpoint="/home/adhemar/Bureau/METHAN/code/logs/train_terra_fcos_iteration_1_2/epoch_72.pth" \
  --output_path="/home/adhemar/Bureau/METHAN/code/results/france_iter_1_1" \
  --batch_size=1 \
  --num_workers=1 \
  --resume=False