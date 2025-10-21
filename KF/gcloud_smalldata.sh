#!/bin/bash
set -euo pipefail
mkdir -p logs

# Experiments 10/21: running fno with limited data
re=500
epochs=500
bsize=100
lr=5e-4
# maxlr=1e-3

# backs : model backbone ("fno" or "deeponet")
last_acts=(0)
backs=("fno")
for last_act in "${last_acts[@]}"; do
  for dim in 1024; do
		for back in "${backs[@]}"; do
	  	for train_traj in 20 50 100; do
				for c in 500 1000; do
					project_tag="c${c}_train${train_traj}"

					sbatch gcloud_single.sh --epochs $epochs --bsize $bsize \
						--branch_conv_channels 64 128 256 512 \
						--output_dim $dim --branch_fc_dims $dim \
						--trunk_hidden_dims $dim $dim $dim $dim \
						--dt 1.0 --Re $re \
						--lr $lr \
						--lam_reg_vol 0.1 \
						--project \
						--diag_Q \
						--c_init $c \
						--discrete_proj \
						--tag "$project_tag" \
						--bsize $bsize \
						--backbone $back \
						--sched "multistep" \
						--data_size $train_traj
					done

				noproject_tag="train${train_traj}"
				sbatch gcloud_single.sh --epochs $epochs --bsize $bsize \
					--branch_conv_channels 64 128 256 512 \
					--output_dim $dim --branch_fc_dims $dim \
					--trunk_hidden_dims $dim $dim $dim $dim \
					--dt 1.0 --Re $re \
					--lr $lr \
					--tag "$noproject_tag" \
					--bsize $bsize \
					--backbone $back \
					--sched "multistep" \
					--data_size $train_traj
	  		done
	  	done
		done
	done