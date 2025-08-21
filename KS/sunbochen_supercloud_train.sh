#!/bin/bash
#SBATCH --job-name Multiple_KSROM_training
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

for mom_1 in 0.80 0.85 0.90 0.95; do
  for mom_2 in 0.990, 0.995, 0.997, 0.999; do
    sbatch train_single_sunbochen.sh \
    --epoch 20000 \
    --output_dim 256 \
    --branch_conv_channels 32 64 128\
    --branch_fc_dims 256 \
    --trunk_hidden_dims 256 256 256 \
    --project \
    --discrete_proj \
    --lam_reg_vol 0.1 \
    --trunk_scale 0.05 \
    --c_init 60.0 \
    --dt 0.2 \
    --diag_Q \
    --momentum_1 $mom_1 \
    --momentum_2 $mom_2 \
    --tag "test_momentum"
  done
done