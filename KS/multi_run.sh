#!/bin/bash
#SBATCH --job-name Multistep_ellip_train_submit


for lam_reg_vol in 1e-3 0.01 0.1 1 10 100 
do
    sbatch run.sh  --epoch 10000 --branch_conv_channels --trunk_scale 0.05 --lam_reg_vol $lam_reg_vol --output_dim 256 --branch_fc_dims 256  --trunk_hidden_dims 256 256 256 --tag "dim256_v2" --project --diag_Q --c_init 30.0 
done

