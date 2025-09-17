#!/bin/bash
#SBATCH --job-name lr_sweep

re=40
epoch=1000
dt=1.0

# sbatch run.sh --epoch $epoch --branch_conv_channels 64 128 256 512 --trunk_scale 1.0 --output_dim 256 --branch_fc_dims 256  --trunk_hidden_dims 256 256 256 --dt 1.0 --tag "dim256" --Re $re
# sbatch run.sh --epoch $epoch --branch_conv_channels 64 128 256 512 --trunk_scale 1.0 --output_dim 256 --branch_fc_dims 256  --trunk_hidden_dims 256 256 256 256 --dt 1.0 --tag "dim256" --Re $re
# sbatch run.sh --epoch $epoch --branch_conv_channels 64 128 256 512 --trunk_scale 1.0 --output_dim 512 --branch_fc_dims 512  --trunk_hidden_dims 512 512 512 --dt 1.0 --tag "dim512" --Re $re
# sbatch run.sh --epoch $epoch --branch_conv_channels 64 128 256 512 --trunk_scale 1.0 --output_dim 512 --branch_fc_dims 512  --trunk_hidden_dims 512 512 512 512 --dt 1.0 --tag "dim512" --Re $re
# sbatch run.sh --epoch $epoch --branch_conv_channels 64 128 256 512 --trunk_scale 1.0 --output_dim 1024 --branch_fc_dims 1024  --trunk_hidden_dims 1024 1024 1024 --dt 1.0 --tag "dim1024" --Re $re

# for c in 100 125 200; do
#     sbatch run.sh \
#     --epoch 2000 \
#     --branch_conv_channels 64 128 256 512 \
#     --trunk_scale 1.0 \
#     --output_dim 1024 \
#     --branch_fc_dims 1024 \
#     --trunk_hidden_dims 1024 1024 1024 1024 \
#     --dt 1.0 \
#     --tag "dim1024" \
#     --Re $re \
#     --lr 1e-4 \
#     --lam_reg_vol 0.1 \
#     --project \
#     --diag_Q \
#     --c_init $c \
#     --discrete_proj \
#     --bsize 500
# done 


# for lr in 1e-5 2e-5; do 
#     for c in 250 240 300; do
#         sbatch run.sh \
#         --epoch $epoch \
#         --branch_conv_channels 64 128 256 512 \
#         --trunk_scale 1.0 \
#         --output_dim 1024 \
#         --branch_fc_dims 1024 \
#         --trunk_hidden_dims 1024 1024 1024 1024 \
#         --dt 1.0 \
#         --tag "dim1024" \
#         --Re $re \
#         --lr $lr \
#         --lam_reg_vol 0.1 \
#         --project \
#         --diag_Q \
#         --c_init $c \
#         --discrete_proj \
#         --bsize 2048
#     done 
# done

sbatch run.sh \
--epoch $epoch \
--branch_conv_channels 64 128 256 512 \
--trunk_scale 1.0 \
--output_dim 1024 \
--branch_fc_dims 1024 \
--trunk_hidden_dims 1024 1024 1024 1024 \
--dt $dt \
--tag "dim1024" \
--Re $re \
--lr 1e-4 \
--bsize 500


for c in 300; do
    sbatch run.sh \
    --epoch $epoch \
    --branch_conv_channels 64 128 256 512 \
    --trunk_scale 1.0 \
    --output_dim 1024 \
    --branch_fc_dims 1024 \
    --trunk_hidden_dims 1024 1024 1024 1024 \
    --dt $dt \
    --tag "dim1024" \
    --Re $re \
    --lr 1e-4 \
    --lam_reg_vol 0.1 \
    --project \
    --diag_Q \
    --c_init $c \
    --discrete_proj \
    --bsize 500
done 



# --lam_reg_vol 0.1 \
# --project \
# --diag_Q \
# --c_init $c \
# --discrete_proj \
# --bsize 500
 



# for c in 100.0 150.0 200.0
# do
#     sbatch run.sh --epoch 10000 --branch_conv_channels 64 128 256 512 --trunk_scale 1.0 --output_dim 256 --branch_fc_dims 256  --trunk_hidden_dims 256 256 256 --lam_reg_vol 0.1 --project --diag_Q --c_init $c --dt 1.0 --discrete_proj
# done

