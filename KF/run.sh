#!/bin/bash

#SBATCH -c 5 --gres=gpu:volta:1

# Loading the required module
source /etc/profile
# module load anaconda/2023a-pytorch
module load anaconda/Python-ML-2025a

# Run the script
python -u train.py "$@"