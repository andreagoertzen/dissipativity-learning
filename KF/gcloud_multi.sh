#!/bin/bash
set -euo pipefail
mkdir -p logs

re=500
epochs=3000
bsize=512

# New experiments 09/18: lr sweep for Re=500, 128 by 128, with projection
for lr in 1e-3 5e-4 2e-4 1e-4 5e-5; do
  for dim in 1024 2048; do
    for lam in 1e-3 1e-1; do
        tag="lr${lr}_dim${dim}_lam${lam}"
        sbatch gcloud_single.sh --epochs $epochs --bsize $bsize \
        --branch_conv_channels 64 128 256 512 \
        --output_dim $dim --branch_fc_dims $dim \
        --trunk_hidden_dims $dim $dim $dim \
        --dt 1.0 --Re $re \
        --lr $lr --circular_padding \
        --tag "$tag" \
        --project --discrete_proj --diag_Q --c_init 500.0 --lam_reg_vol $lam
        done
    done
done