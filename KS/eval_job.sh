#!/bin/bash

#SBATCH --gres=gpu:volta:1
#SBATCH --exclusive -c 1

# Loading the required module
source /etc/profile
module load anaconda/2023a-pytorch

MODEL_DIR="$1"
shift

python -u eval.py --model_dir "${MODEL_DIR}" "$@"
