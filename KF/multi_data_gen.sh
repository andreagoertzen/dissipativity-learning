#!/usr/bin/env bash
#SBATCH -J kflow
#SBATCH -p cpu-gpu-rtx8000
#SBATCH --gres=gpu:1                 # one GPU per job
# If your site requires typed GRES, use instead:
# SBATCH --gres=gpu:rtx8000:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 3-00:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -euo pipefail

# (optional) conda env
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate pde_torch_env
fi

# Two ic_factors — one per task
IC_FACTORS=(30.0 100.0)
IC="${IC_FACTORS[$SLURM_ARRAY_TASK_ID]}"
DT_FACTORS=(0.0001 0.00004)
DT="${DT_FACTORS[$SLURM_ARRAY_TASK_ID]}"

# Your other knobs (edit if you like)
RE=500
TT=500
GRID=128
TSAVE=1.0
N_TRAJ=200

echo "[info] task ${SLURM_ARRAY_TASK_ID} on $(hostname) using ic_factor=${IC}"
srun python KF_data_gen.py \
  --Re "${RE}" \
  --dt "${DT}" \
  --T "${TT}" \
  --grid_size "${GRID}" \
  --tsave "${TSAVE}" \
  --n_traj "${N_TRAJ}" \
  --ic_factor "${IC}"


# When running this, use an array: sbatch --array=0-1 multi_data_gen.sh