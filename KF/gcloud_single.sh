#!/bin/bash
#SBATCH --job-name=train
#SBATCH -o logs/%x_%j.out
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00

# ---- conda activation (no modules) ----
source /home/tangsun_mit_edu/miniconda3/etc/profile.d/conda.sh
conda activate chaos_env

# ---- headless matplotlib for cluster ----
export MPLBACKEND=Agg

# ---- run training; pass sbatch args through ----
python -u train.py "$@"
