#!/bin/bash
#SBATCH --job-name lam_lr_sweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

for lr in 1e-4 2e-5; do
  for lam in 1.0 0.1 0.01 0.02; do
    sbatch train_single_sunbochen.sh \
    --epoch 10000 \
    --c_init 40.0 \
    --discrete-proj \
    --lam-reg-vol $lam
  done
done