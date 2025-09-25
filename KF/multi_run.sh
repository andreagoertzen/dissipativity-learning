#!/bin/bash
#SBATCH --job-name lr_dim_sweep

re=40
epochs=3000
bsize=512
# maxlr=1e-3

# Experiments 09/20: larger model, ablation on last layer activation
# No projection, with projection is running on google cloud
# Run both variants: 0 = no flag, 1 = include --trunk_last_act
last_acts=(1)
for lr in 1e-4; do
  for dim in 1024; do
    for c in 240 250 300; do
      tag="lr${lr}_dim${dim}_lam${lam}_act${last_act}"

      cmd="sbatch run.sh --epochs $epochs --bsize $bsize \
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

# # New experiments 09/17: lr sweep for Re=500, 128 by 128
# for lr in 1e-3 5e-4 2e-4 1e-4; do
#   for dim in 1024 2048; do
#     tag="lr${lr}_dim${dim}"
#     sbatch run.sh --epochs $epochs --bsize $bsize \
#       --branch_conv_channels 64 128 256 512 \
#       --output_dim $dim --branch_fc_dims $dim \
#       --trunk_hidden_dims $dim $dim $dim \
#       --dt 1.0 --Re $re \
#       --lr $lr --circular_padding \
#       --tag "$tag"
#     done
# done

#     --lam_reg_vol 0.1 \
#     --project \
#     --diag_Q \
#     --c_init $c \
#     --discrete_proj \

# for lr in 1e-4; do
#   for dim in 1024; do
#     tag="dim${dim}_trunkact"
#     sbatch run.sh --epochs $epochs --bsize $bsize \
#       --branch_conv_channels 64 128 256 512 \
#       --output_dim $dim --branch_fc_dims $dim \
#       --trunk_hidden_dims $dim $dim $dim \
#       --dt 0.5 --Re $re \
#       --lr $lr \
#       --tag "$tag"
#     done
# done

# Previous scripts for running 64 by 64 experiments
# for norm in 0 1; do
#   for circ in 0 1; do
#     for sched in cosine multistep; do
#       tag="Abl1_norm${norm}_circ${circ}_sched${sched}"
#       sbatch run.sh --epochs $epochs --bsize $bsize \
#         --branch_conv_channels 64 128 256 512 \
#         --output_dim 256 --branch_fc_dims 256 \
#         --trunk_hidden_dims 256 256 256 \
#         --dt 1.0 --Re $re \
#         --lr $maxlr --sched $sched --warmup_epochs 100 --min_lr 1e-5 --gamma 0.5 \
#         $( [ $norm -eq 1 ] && echo --normalize ) \
#         $( [ $circ -eq 1 ] && echo --circular_padding ) \
#         --clip_grad 1.0 \
#         --tag "$tag"
#     done
#   done
# done


# re=500
# epoch=4000

# for lr in 2e-3 1e-3 5e-4 2e-4
# do
#     # sbatch run.sh --epoch $epoch --branch_conv_channels 64 128 256 512 --trunk_scale 1.0 --output_dim 256 --branch_fc_dims 256  --trunk_hidden_dims 256 256 256 --dt 1.0 --tag "dim256" --Re $re --lr $lr --scheduler --activation SiLU --circular_padding
#     # sbatch run.sh --epoch $epoch --branch_conv_channels 64 128 256 512 --trunk_scale 1.0 --output_dim 256 --branch_fc_dims 256  --trunk_hidden_dims 256 256 256 256 --dt 1.0 --tag "dim256" --Re $re --lr $lr --activation SiLU --circular_padding
#     sbatch run.sh --epoch $epoch --branch_conv_channels 64 128 256 512 --trunk_scale 1.0 --output_dim 512 --branch_fc_dims 512  --trunk_hidden_dims 512 512 512 --dt 1.0 --tag "dim512" --Re $re --lr $lr --scheduler --activation SiLU --circular_padding
#     sbatch run.sh --epoch $epoch --branch_conv_channels 64 128 256 512 --trunk_scale 1.0 --output_dim 512 --branch_fc_dims 512  --trunk_hidden_dims 512 512 512 512 --dt 1.0 --tag "dim512" --Re $re --lr $lr --activation SiLU --circular_padding
# done

# sbatch run.sh --epoch $epoch --branch_conv_channels 64 128 256 512 --trunk_scale 1.0 --output_dim 1024 --branch_fc_dims 1024  --trunk_hidden_dims 1024 1024 1024 --dt 1.0 --tag "dim1024" --Re $re
# sbatch run.sh --epoch $epoch --branch_conv_channels 64 128 256 512 --trunk_scale 1.0 --output_dim 1024 --branch_fc_dims 1024  --trunk_hidden_dims 1024 1024 1024 1024 --dt 1.0 --tag "dim1024" --Re $re

# Andrea's code for Re=40
# re=40
# epoch=1000
# dt=1.0

# # sbatch run.sh --epoch $epoch --branch_conv_channels 64 128 256 512 --trunk_scale 1.0 --output_dim 256 --branch_fc_dims 256  --trunk_hidden_dims 256 256 256 --dt 1.0 --tag "dim256" --Re $re
# # sbatch run.sh --epoch $epoch --branch_conv_channels 64 128 256 512 --trunk_scale 1.0 --output_dim 256 --branch_fc_dims 256  --trunk_hidden_dims 256 256 256 256 --dt 1.0 --tag "dim256" --Re $re
# # sbatch run.sh --epoch $epoch --branch_conv_channels 64 128 256 512 --trunk_scale 1.0 --output_dim 512 --branch_fc_dims 512  --trunk_hidden_dims 512 512 512 --dt 1.0 --tag "dim512" --Re $re
# # sbatch run.sh --epoch $epoch --branch_conv_channels 64 128 256 512 --trunk_scale 1.0 --output_dim 512 --branch_fc_dims 512  --trunk_hidden_dims 512 512 512 512 --dt 1.0 --tag "dim512" --Re $re
# # sbatch run.sh --epoch $epoch --branch_conv_channels 64 128 256 512 --trunk_scale 1.0 --output_dim 1024 --branch_fc_dims 1024  --trunk_hidden_dims 1024 1024 1024 --dt 1.0 --tag "dim1024" --Re $re

# # for c in 100 125 200; do
# #     sbatch run.sh \
# #     --epoch 2000 \
# #     --branch_conv_channels 64 128 256 512 \
# #     --trunk_scale 1.0 \
# #     --output_dim 1024 \
# #     --branch_fc_dims 1024 \
# #     --trunk_hidden_dims 1024 1024 1024 1024 \
# #     --dt 1.0 \
# #     --tag "dim1024" \
# #     --Re $re \
# #     --lr 1e-4 \
# #     --lam_reg_vol 0.1 \
# #     --project \
# #     --diag_Q \
# #     --c_init $c \
# #     --discrete_proj \
# #     --bsize 500
# # done 


# # for lr in 1e-5 2e-5; do 
# #     for c in 250 240 300; do
# #         sbatch run.sh \
# #         --epoch $epoch \
# #         --branch_conv_channels 64 128 256 512 \
# #         --trunk_scale 1.0 \
# #         --output_dim 1024 \
# #         --branch_fc_dims 1024 \
# #         --trunk_hidden_dims 1024 1024 1024 1024 \
# #         --dt 1.0 \
# #         --tag "dim1024" \
# #         --Re $re \
# #         --lr $lr \
# #         --lam_reg_vol 0.1 \
# #         --project \
# #         --diag_Q \
# #         --c_init $c \
# #         --discrete_proj \
# #         --bsize 2048
# #     done 
# # done

# sbatch run.sh \
# --epoch 1000 \
# --branch_conv_channels 64 128 256 512 \
# --trunk_scale 1.0 \
# --output_dim 1024 \
# --branch_fc_dims 1024 \
# --trunk_hidden_dims 1024 1024 1024 1024 \
# --dt 1.0 \
# --tag "dim1024_b" \
# --Re 40 \
# --lr 1e-4 \
# --bsize 500


# for c in 240 250 300; do
#     sbatch run.sh \
#     --epoch 1000 \
#     --branch_conv_channels 64 128 256 512 \
#     --trunk_scale 1.0 \
#     --output_dim 1024 \
#     --branch_fc_dims 1024 \
#     --trunk_hidden_dims 1024 1024 1024 1024 \
#     --dt 1.0 \
#     --tag "dim1024_b" \
#     --Re 40 \
#     --lr 1e-4 \
#     --lam_reg_vol 0.1 \
#     --project \
#     --diag_Q \
#     --c_init $c \
#     --discrete_proj \
#     --bsize 500
# done 



# # --lam_reg_vol 0.1 \
# # --project \
# # --diag_Q \
# # --c_init $c \
# # --discrete_proj \
# # --bsize 500
 



# # for c in 100.0 150.0 200.0
# # do
# #     sbatch run.sh --epoch 10000 --branch_conv_channels 64 128 256 512 --trunk_scale 1.0 --output_dim 256 --branch_fc_dims 256  --trunk_hidden_dims 256 256 256 --lam_reg_vol 0.1 --project --diag_Q --c_init $c --dt 1.0 --discrete_proj
# # done