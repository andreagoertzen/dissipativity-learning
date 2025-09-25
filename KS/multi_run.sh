#!/bin/bash
#SBATCH --job-name Multistep_ellip_train_submit


# sbatch run.sh --epoch 40000 --branch_conv_channels 32 64 128 --trunk_scale 0.05 --output_dim 256 --branch_fc_dims 256  --trunk_hidden_dims 256 256 256 --tag "dim256_train2" --dt 1.0
for c in 50.0 55.0 60.0 
do
    sbatch run.sh --epoch 40000 --branch_conv_channels 32 64 128 --trunk_scale 0.05 --output_dim 256 --branch_fc_dims 256  --trunk_hidden_dims 256 256 256 --tag "dim256_train2" --lam_reg_vol 0.1 --project --diag_Q --c_init $c --dt 1.0 --discrete_proj
done

# for c in 45.0 60.0 100.0
# do
#     sbatch run.sh --epoch 70000 --branch_conv_channels 32 64 128 256 --trunk_scale 0.05 --output_dim 256 --branch_fc_dims 256  --trunk_hidden_dims 256 256 256 --tag "dim256" --lam_reg_vol 0.1 --project --diag_Q --c_init $c --dt 1.0 --discrete_proj
# done
# sbatch run.sh --epoch 70000 --branch_conv_channels 32 64 128 256 --trunk_scale 0.05 --output_dim 256 --branch_fc_dims 256  --trunk_hidden_dims 256 256 256 --tag "dim256" --dt 1.0

# for c in 45.0 60.0 100.0
# do
#     sbatch run.sh --epoch 40000 --branch_conv_channels 32 64 128 256 --trunk_scale 0.05 --output_dim 512 --branch_fc_dims 512  --trunk_hidden_dims 512 512 512 --tag "dim512" --lam_reg_vol 0.1 --project --diag_Q --c_init $c --dt 1.0 --discrete_proj
# done
# sbatch run.sh --epoch 40000 --branch_conv_channels 32 64 128 256 --trunk_scale 0.05 --output_dim 512 --branch_fc_dims 512  --trunk_hidden_dims 512 512 512 --tag "dim512" --dt 1.0

# sbatch run.sh --epoch 10000 --output_dim 256 --branch_fc_dims 256  --trunk_hidden_dims 256 256 256 --tag "dim256"
# sbatch run.sh --epoch 40000 --trunk_scale 0.05 --output_dim 256 --branch_fc_dims 256  --trunk_hidden_dims 256 256 256 --tag "dim256"
# sbatch run.sh --epoch 40000 --branch_conv_channels 32 64 128 256 --trunk_scale 0.05 --output_dim 256 --branch_fc_dims 256  --trunk_hidden_dims 256 256 256 --tag "dim256"
# sbatch run.sh --epoch 10000 --branch_conv_channels --output_dim 256 --branch_fc_dims 256  --trunk_hidden_dims 256 256 256 --tag "dim256"
# sbatch run.sh --epoch 10000 --branch_conv_channels --trunk_scale 0.05 --output_dim 256 --branch_fc_dims 256  --trunk_hidden_dims 256 256 256 --tag "dim256"
