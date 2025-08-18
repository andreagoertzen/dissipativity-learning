#!/bin/bash
#SBATCH --job-name Multistep_ellip_train_submit


for lam_reg_vol in 1 
do
    sbatch run.sh  --epoch 20000 --branch_conv_channels --trunk_scale 0.05 --lam_reg_vol $lam_reg_vol --output_dim 256 --branch_fc_dims 256  --trunk_hidden_dims 256 256 256 --tag "dim256_printgrad" --project --diag_Q --trainable_c --c_init 1.0 
done

