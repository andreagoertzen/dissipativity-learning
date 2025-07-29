#!/bin/bash

#SBATCH --gres=gpu:volta:1
#SBATCH --exclusive -c 40

# Loading the required module
source /etc/profile
module load anaconda/2022b

python -u  NSE_training_data.py
