#!/bin/bash
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00

set -euo pipefail

# --- Safety check for arguments ---
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: sbatch $0 <path_to_param_file> <re_number> [cosine_eval_steps]"
    exit 1
fi

PARAM_FILE="$1"
RE_NUMBER="$2"
COS_STEPS="${3:-}"

# ---- Conda Activation ----
source /home/tangsun_mit_edu/miniconda3/etc/profile.d/conda.sh
conda activate chaos_env

# ---- Headless Matplotlib for Cluster ----
export MPLBACKEND=Agg

echo "========================================================"
echo "Starting evaluation"
echo "  SLURM Job ID : ${SLURM_JOB_ID:-N/A}"
echo "  PARAM_FILE   : $PARAM_FILE"
echo "  RE_NUMBER    : $RE_NUMBER"
echo "  COS_STEPS    : ${COS_STEPS:-<none>}"
echo "========================================================"

# ---- Execute Python Script ----
if [ -n "$COS_STEPS" ]; then
    python -u eval.py "$PARAM_FILE" "$RE_NUMBER" "$COS_STEPS"
else
    python -u eval.py "$PARAM_FILE" "$RE_NUMBER"
fi

echo "========================================================"
echo "Evaluation finished."
echo "========================================================"
