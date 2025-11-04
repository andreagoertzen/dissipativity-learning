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
IC_FACTORS=(30.0 30.0 30.0)
IC="${IC_FACTORS[$SLURM_ARRAY_TASK_ID]}"
DT_FACTORS=(0.0005 0.00025 0.0001)
DT="${DT_FACTORS[$SLURM_ARRAY_TASK_ID]}"

# Basic validation: ensure the array index is in range. This makes it
# easier to run with larger arrays (e.g. sbatch --array=0-2) and gives a
# clear error if you accidentally submit the wrong range.
NUM_TASKS=${#IC_FACTORS[@]}
if [ -z "${SLURM_ARRAY_TASK_ID+x}" ]; then
  echo "[error] SLURM_ARRAY_TASK_ID not set. Submit with sbatch --array=0-$((NUM_TASKS-1)) multi_data_gen.sh"
  exit 1
fi
if [ "$SLURM_ARRAY_TASK_ID" -lt 0 ] || [ "$SLURM_ARRAY_TASK_ID" -ge "$NUM_TASKS" ]; then
  echo "[error] SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range (0..$((NUM_TASKS-1)))."
  exit 1
fi

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


# When running this, use an array whose indices match the length of
# the IC_FACTORS/DT_FACTORS arrays above. For example, there are 3
# entries so use 0-2:
#   sbatch --array=0-2 multi_data_gen.sh
# To limit concurrency (max tasks running at once) add a %N suffix,
# for example run at most 3 simultaneously:
#   sbatch --array=0-2%3 multi_data_gen.sh
# You can also set the array directive inside this script (uncomment
# and edit) to make the range permanent:
#   #SBATCH --array=0-2%3