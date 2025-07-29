#!/bin/bash

#SBATCH --gres=gpu:volta:1
#SBATCH --exclusive -c 40

# Loading the required module
source /etc/profile
module load anaconda/2023a-pytorch

# Run the script
python -u NSE_FNO.py 
python -u infani_v2.py
python -u statgen.py
python -u statgen2.py
