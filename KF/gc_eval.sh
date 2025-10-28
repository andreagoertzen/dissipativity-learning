#!/bin/bash
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00

# --- Safety check for arguments ---
if [ "$#" -ne 2 ]; then
    echo "Error: Incorrect number of arguments provided."
    echo "Usage: sbatch $0 <path_to_param_file> <re_number>"
    exit 1
fi

PARAM_FILE="$1"
RE_NUMBER="$2"

# --- Dynamic job and log naming based on arguments ---
#SBATCH --job-name=eval_Re_${RE_NUMBER}
#SBATCH -o logs/eval_Re_${RE_NUMBER}_%j.out

# ---- Conda Activation ----
source /home/tangsun_mit_edu/miniconda3/etc/profile.d/conda.sh
conda activate chaos_env

# ---- Headless Matplotlib for Cluster ----
export MPLBACKEND=Agg

echo "========================================================"
echo "Starting evaluation for Re = $RE_NUMBER"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Parameter file: $PARAM_FILE"
echo "========================================================"

# ---- Execute Python Script ----
# Using named arguments is often clearer if you modify eval.py
# For now, we'll stick to passing them as positional arguments.
python -u eval.py "$PARAM_FILE" "$RE_NUMBER"

echo "========================================================"
echo "Evaluation finished for Re = $RE_NUMBER."
echo "========================================================"