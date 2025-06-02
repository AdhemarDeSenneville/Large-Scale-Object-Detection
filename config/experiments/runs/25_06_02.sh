python /home/adhemar/Bureau/METHAN/code/githubs/Large-Selective-Kernel-Network/tools/train.py\
   /home/adhemar/Bureau/METHAN/code/config/experiments/lsk/rotated_fcos_r50_fpn_1x_dota_le90.py\
      --work-dir /home/adhemar/Bureau/METHAN/code/logs/train_bdortho_150cm_01\
        --cfg-options\
             data.train.ann_file=/home/adhemar/Bureau/datasets/Methanizers/V_multy_source/res_1.5/label/train\
             data.train.img_prefix=/home/adhemar/Bureau/datasets/Methanizers/V_multy_source/res_1.5/image/BDORTHO/train\
             data.val.ann_file=/home/adhemar/Bureau/datasets/Methanizers/V_multy_source/res_1.5/label/val\
             data.val.img_prefix=/home/adhemar/Bureau/datasets/Methanizers/V_multy_source/res_1.5/image/BDORTHO/val\
             data.test.ann_file=/home/adhemar/Bureau/datasets/Methanizers/V_multy_source/res_1.5/label/val\
             data.test.img_prefix=/home/adhemar/Bureau/datasets/Methanizers/V_multy_source/res_1.5/image/BDORTHO/val\