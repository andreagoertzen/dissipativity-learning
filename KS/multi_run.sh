#!/bin/bash
#SBATCH --job-name Multistep_ellip_train_submit


for c in 30.0 35.0 40.0 45.0
do
    sbatch run.sh --epoch 40000 --branch_conv_channels 32 64 128 256 --trunk_scale 0.05 --output_dim 256 --branch_fc_dims 256  --trunk_hidden_dims 256 256 256 --tag "dim256" --lam_reg_vol 0.1 --project --diag_Q --c_init $c --dt 0.2 --discrete_proj --warm_start
done

# sbatch run.sh --epoch 10000 --output_dim 256 --branch_fc_dims 256  --trunk_hidden_dims 256 256 256 --tag "dim256"
# sbatch run.sh --epoch 40000 --trunk_scale 0.05 --output_dim 256 --branch_fc_dims 256  --trunk_hidden_dims 256 256 256 --tag "dim256"
# sbatch run.sh --epoch 40000 --branch_conv_channels 32 64 128 256 --trunk_scale 0.05 --output_dim 256 --branch_fc_dims 256  --trunk_hidden_dims 256 256 256 --tag "dim256"
# sbatch run.sh --epoch 10000 --branch_conv_channels --output_dim 256 --branch_fc_dims 256  --trunk_hidden_dims 256 256 256 --tag "dim256"
# sbatch run.sh --epoch 10000 --branch_conv_channels --trunk_scale 0.05 --output_dim 256 --branch_fc_dims 256  --trunk_hidden_dims 256 256 256 --tag "dim256"
