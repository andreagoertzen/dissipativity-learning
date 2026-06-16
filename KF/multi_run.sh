#!/bin/bash
#SBATCH --job-name backbone_sweep

# re=500
# epochs=50
# bsize=50
# lr=5e-4
# # maxlr=1e-3

# # backs : model backbone ("fno" or "deeponet")
# last_acts=(0)
# backs=("fno")
# for last_act in "${last_acts[@]}"; do
#   for n_traj in 200; do
#     for lam_reg in 0.1 0.01; do
#       for c in 400 450 500; do
#         for back in "${backs[@]}"; do
#           # tag="dim${dim}_act${last_act}"

#           cmd="sbatch run.sh --epochs $epochs --bsize $bsize \
#             --sched multistep \
#             --n_traj $n_traj \
#             --dt 1.0 --Re $re \
#             --lr $lr \
#             --tag ICscale_1 \
#             --lam_reg_vol $lam_reg \
#             --project \
#             --c_init $c \
#             --discrete_proj \
#             --bsize $bsize \
#             --backbone $back \
#             --nn_Q \
#             --nn_x0"

#           # Conditionally add the flag
#           if [ "$last_act" -eq 1 ]; then
#             cmd="$cmd --trunk_last_act"
#           fi

#           echo "$cmd"
#           eval "$cmd"
#         done
#       done
#     done
#   done
# done


# re=40
# epochs=50 # note that successful one does 2000
# bsize=50 # note that successful one does 500
# lr=5e-4 # note that successful one does 1e-4 and no scheduler
# # maxlr=1e-3

# # backs : model backbone ("fno" or "deeponet")
# last_acts=(0)
# backs=("fno")
# for last_act in "${last_acts[@]}"; do
#   for n_traj in 200; do
#     for lam_reg in 0.1 0.001; do
#       for c in 200 300 400; do
#         for back in "${backs[@]}"; do
#           # tag="dim${dim}_act${last_act}"

#           cmd="sbatch run.sh --epochs $epochs --bsize $bsize \
#             --sched multistep \
#             --n_traj $n_traj \
#             --dt 1.0 --Re $re \
#             --lr $lr \
#             --tag ICscale_1 \
#             --lam_reg_vol $lam_reg \
#             --project \
#             --c_init $c \
#             --discrete_proj \
#             --bsize $bsize \
#             --backbone $back \
#             --nn_Q \
#             --nn_x0"

#           # Conditionally add the flag
#           if [ "$last_act" -eq 1 ]; then
#             cmd="$cmd --trunk_last_act"
#           fi

#           echo "$cmd"
#           eval "$cmd"
#         done
#       done
#     done
#   done
# done



# re=500
# epochs=50 # note that successful one does 2000
# bsize=50 # note that successful one does 500
# lr=5e-5 # note that successful one does 1e-4 and no scheduler
# # maxlr=1e-3

# # backs : model backbone ("fno" or "deeponet")
# last_acts=(0)
# backs=("fno")
# for n_traj in 200; do
#   for back in "${backs[@]}"; do
#     # tag="dim${dim}_act${last_act}"

#     cmd="sbatch run.sh --epochs $epochs --bsize $bsize \
#       --sched multistep \
#       --n_traj $n_traj \
#       --dt 1.0 --Re $re \
#       --lr $lr \
#       --tag ICscale_1 \
#       --backbone $back"

#     echo "$cmd"
#     eval "$cmd"
#   done
# done

for lr in 1e-4; do 
    for c in 275 280 285; do
        sbatch run.sh \
        --epoch 2000 \
        --branch_conv_channels 64 128 256 512 \
        --trunk_scale 1.0 \
        --output_dim 1024 \
        --branch_fc_dims 1024 \
        --trunk_hidden_dims 1024 1024 1024 1024 \
        --dt 1.0 \
        --backbone "deeponet" \
        --tag "dim1024" \
        --Re 40 \
        --lr $lr \
        --lam_reg_vol 0.1 \
        --project \
        --diag_Q \
        --c_init $c \
        --bsize 500 \
        --nn_Q \
        --nn_x0
    done 
done


# for lr in 1e-4; do 
#     sbatch run.sh \
#     --epoch 2000 \
#     --branch_conv_channels 64 128 256 512 \
#     --trunk_scale 1.0 \
#     --output_dim 1024 \
#     --branch_fc_dims 1024 \
#     --trunk_hidden_dims 1024 1024 1024 1024 \
#     --dt 1.0 \
#     --backbone "deeponet" \
#     --tag "dim1024" \
#     --Re 40 \
#     --lr $lr \
#     --bsize 500
# done
