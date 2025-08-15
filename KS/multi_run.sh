#!/bin/bash
#SBATCH --job-name Multistep_ellip_train_submit


for lam_reg_vol in 1000000 100000 10000 1000 100 1 0.5 0.1 0.01
do
    sbatch run.sh --project True --trunk_scale 0.05 --lam_reg_vol $lam_reg_vol
done