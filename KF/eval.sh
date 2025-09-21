#!/bin/bash

#SBATCH -c 5 --gres=gpu:volta:1

# Loading the required module
source /etc/profile
module load anaconda/2023a-pytorch

# Run the script
python -u eval.py "$@"