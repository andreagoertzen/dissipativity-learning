#!/bin/bash

#SBATCH --gres=gpu:volta:1
#SBATCH --exclusive -c 40

# Loading the required module
source /etc/profile
module load anaconda/2023a-pytorch

# Run the script
python -u train2.py 