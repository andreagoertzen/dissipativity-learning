#!/bin/bash
#SBATCH --job-name gen_KF_data
#SBATCH --output=logs/%x_%j.out

sbatch single_data.sh