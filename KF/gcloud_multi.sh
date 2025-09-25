#!/bin/bash
set -euo pipefail
mkdir -p logs

re=40
epochs=3000
bsize=512

last_acts=(1)
for lr in 1e-4; do
  for dim in 1024; do
    for c in 240 250 300; do
      tag="lr${lr}_dim${dim}_act${last_act}"

      cmd="sbatch gcloud_single.sh --epochs $epochs --bsize $bsize \
        --branch_conv_channels 64 128 256 512 \
        --output_dim $dim --branch_fc_dims $dim \
        --trunk_hidden_dims $dim $dim $dim $dim \
        --dt 1.0 --Re $re \
        --lr $lr --circular_padding \
        --tag \"$tag\" \
        --lam_reg_vol 0.1 \
        --project \
        --diag_Q \
        --c_init $c \
        --discrete_proj \
        --bsize 500 "

      # Conditionally add the flag
      if [ "$last_act" -eq 1 ]; then
        cmd="$cmd --trunk_last_act"
      fi

      echo "$cmd"
      eval "$cmd"
    done
  done
done

# re=500
# epochs=3000
# bsize=512

# # Experiments 09/22: larger c, with activation
# for lr in 1e-3 2e-4; do
#   for dim in 1024; do
#     for lam in 1e-1 1e-2 1e-3 1e-4 1e-5; do
#       for c0 in 2000.0 4000.0; do
#         tag="c0_${c0}"

#         cmd="sbatch gcloud_single.sh --epochs $epochs --bsize $bsize \
#           --branch_conv_channels 64 128 256 512 \
#           --output_dim $dim --branch_fc_dims $dim \
#           --trunk_hidden_dims $dim $dim $dim \
#           --dt 0.5 --Re $re \
#           --lr $lr --circular_padding \
#           --tag \"$tag\" \
#           --project --discrete_proj --diag_Q --c_init $c0 --lam_reg_vol $lam \
#           --trunk_last_act"

#         echo "$cmd"
#         eval "$cmd"
#       done
#     done
#   done
# done


# Experiments 09/20: larger model, ablation on last layer activation

# Run both variants: 0 = no flag, 1 = include --trunk_last_act
# last_acts=(0 1)
# for lr in 1e-3 5e-4 2e-4 1e-4; do
#   for dim in 1024 2048; do
#     for lam in 1e-1; do
#       for last_act in "${last_acts[@]}"; do
#         tag="lr${lr}_dim${dim}_lam${lam}_act${last_act}"

#         cmd="sbatch gcloud_single.sh --epochs $epochs --bsize $bsize \
#           --branch_conv_channels 64 128 256 512 \
#           --output_dim $dim --branch_fc_dims $dim \
#           --trunk_hidden_dims $dim $dim $dim $dim \
#           --dt 0.5 --Re $re \
#           --lr $lr --circular_padding \
#           --tag \"$tag\" \
#           --project --discrete_proj --diag_Q --c_init 1000.0 --lam_reg_vol $lam"

#         # Conditionally add the flag
#         if [ "$last_act" -eq 1 ]; then
#           cmd="$cmd --trunk_last_act"
#         fi

#         echo "$cmd"
#         eval "$cmd"
#       done
#     done
#   done
# done

# # New experiments 09/18: lr sweep for Re=500, 128 by 128, with projection
# for lr in 1e-3 5e-4 2e-4 1e-4; do
#   for dim in 1024 2048; do
#     for lam in 1e-1; do
#         tag="lr${lr}_dim${dim}_lam${lam}"
#         sbatch gcloud_single.sh --epochs $epochs --bsize $bsize \
#         --branch_conv_channels 64 128 256 512 \
#         --output_dim $dim --branch_fc_dims $dim \
#         --trunk_hidden_dims $dim $dim $dim $dim \
#         --dt 1.0 --Re $re \
#         --lr $lr --circular_padding \
#         --tag "$tag" \
#         --project --discrete_proj --diag_Q --c_init 1000.0 --lam_reg_vol $lam
#         done
#     done
# done