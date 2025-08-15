#!/bin/bash
#SBATCH --job-name Multistep_ellip_eval_submit


for lam_reg_vol in 10 1 0.1 0.01
do
    sbatch eval.sh --project True --trunk_scale 0.05 --lam_reg_vol $lam_reg_vol --epochs 2000
done