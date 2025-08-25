#!/bin/bash
#SBATCH --job-name lam_lr_sweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Last batch of experiments seem to show somehow lr=5e-5 is the sweet spot for training unconstrained models

for lr in 5e-5; do
  for lam in 10.0 1.0 0.1 0.01 1e-3 1e-4; do
    sbatch train_single_sunbochen.sh \
    --epoch 4000 \
    --c-init 60.0 \
    --discrete-proj \
    --lam-reg-vol $lam \
    --lr $lr
  done
done

# for lr in 5e-5 1e-5; do
#   sbatch train_single_sunbochen.sh \
#     --epoch 10000 \
#     --lr $lr
# done