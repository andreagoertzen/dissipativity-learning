#!/bin/bash
#SBATCH --job-name lam_lr_sweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# 0824_17/23: Try a very long trajectory dataset, like Anima's paper suggested
# 0824_17/22: Last batch of experiments seem to show somehow lr=5e-5 is the sweet spot for training unconstrained models; however, the model predictions don't turn out to be doing well

for lr in 5e-5; do
  for lam in 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9; do
    sbatch train_single_sunbochen.sh \
    --epoch 4000 \
    --c-init 20.0 \
    --discrete-proj \
    --lam-reg-vol $lam \
    --lr $lr \
    --trainable_c
  done
done

# for lr in 5e-5; do
#   sbatch train_single_sunbochen.sh \
#     --epoch 4000 \
#     --lr $lr
# done