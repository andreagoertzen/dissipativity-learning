#!/bin/bash
#SBATCH --job-name=deeponet
#SBATCH -o logs/%x_%j.out
#SBATCH --partition=cpu-gpu-v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00

# ---- conda activation (no modules) ----
source /mnt/home/tangsun/miniconda3/etc/profile.d/conda.sh
conda activate pde_torch_env

# ---- headless matplotlib for cluster ----
export MPLBACKEND=Agg

# ---- run training; pass sbatch args through ----
python -u train.py "$@"
