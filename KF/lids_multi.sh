#!/bin/bash
set -euo pipefail
mkdir -p logs

re=500
epochs=3000
bsize=512
maxlr=1e-3

norms=(0 1)
circs=(0 1)
scheds=(cosine multistep)

for norm in "${norms[@]}"; do
  for circ in "${circs[@]}"; do
    for sched in "${scheds[@]}"; do
      for lam in 1e-3 1e-1; do
        tag="lids_Abl1_norm${norm}_circ${circ}_sched${sched}_proj"
        sbatch lids_single.sh \
            --epochs $epochs --bsize $bsize \
            --branch_conv_channels 64 128 256 512 \
            --output_dim 256 --branch_fc_dims 256 \
            --trunk_hidden_dims 256 256 256 \
            --dt 1.0 --Re $re \
            --lr $maxlr --sched $sched --min_lr 1e-5 --gamma 0.5 \
            $( [ $norm -eq 1 ] && echo --normalize ) \
            $( [ $circ -eq 1 ] && echo --circular_padding ) \
            --clip_grad 1.0 \
            --tag "$tag" \
            --project --discrete_proj --diag_Q --lam_reg_vol 1e-3 --c_init 300.0
        done
    done
  done
done
