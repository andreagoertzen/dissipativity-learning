#!/bin/bash

#SBATCH --gres=gpu:volta:1

# Loading the required module
source /etc/profile
module load anaconda/2023a-pytorch

# Run the script
python -u KF_data_gen.py
