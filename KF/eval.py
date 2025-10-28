import torch
from utils import (
    animate_rollout,
    one_step_animation,
    rollout_animation,
    pca_modes,
    visualize_ellipsoid,
    compare_distributions,
    pca_histogram_eval,
    evaluate_fourier_spectrum,
    spatial_corr,
    fourier_spectrum_2d,
    energy_time,
    rollout_trajectory_batched,
    animate_saved_rollout,
    plot_cos_sims,
    sinkhorn_div,
    covariance_rmse,
    time_correlation_metric,
)
from model import ECO
import sys
from pathlib import Path
import os
import numpy as np

def run_functions(params, param_path_parent, Re, test_idx=0, cosine_eval_steps=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trunk_scale = 1
    m = 64
    n = 2
    model_folder = param_path_parent
    print(model_folder)
    figs_dir = figs_folder = f'{model_folder}/eval_results'
    if not os.path.exists(figs_folder):
        os.makedirs(figs_folder)

    # for key, value in params.items():
    #     print(f"{key}: {value}")

    model_params = {
        'm': m,
        'n': n,
        'trainable_c': params['trainable_c'],
        'c0': params['c0'],
        'project': params['project'],
        'diag_Q': params['diag_Q'],
        'branch_conv_channels': params['branch_conv_channels'].tolist(),
        'branch_fc_dims': params['branch_fc_dims'].tolist(),
        'trunk_hidden_dims': params['trunk_hidden_dims'].tolist(),
        'output_dim': params['output_dim'],
        'dt': params['dt'],
        'discrete_proj': params['discrete_proj'],
        'circular_padding': params['circular_padding'],
        'trunk_last_act': params['trunk_last_act'],
        'backbone': params['backbone']
        # 'backbone': 'deeponet',
    }

    model = ECO(model_params).to(device)
    print(next(model.parameters()).is_cuda)
    num_params = sum(v.numel() for v in model.parameters() if v.requires_grad)
    print(f'model params: {num_params}')


    model.load_state_dict(torch.load(f'{model_folder}/model_epoch_best.pt',map_location=device))
    model.eval()

    ## GET MODEL PARAMETERS
    if model.project:
        Q = torch.diag(model.V._construct_Q()).detach().cpu().numpy()
        c = model.c.detach().cpu().numpy()
    else:
        Q = None
        c = 30.0

    ### LOAD DATA
    print('LOADING TEST DATA')
    # file_dir = f'data/KF_Re{Re}_M64_tsave1_T5000_n1/data.pt'
    # file_dir = f'data/KF_Re{Re}_M64_tsave1_T500_n200/data.pt'
    # file_dir = f'data/KF_Re{Re}_M128_tsave0.5_T5000_n1/data.pt'
    # file_dir = f'data/KF_Re{Re}_M128_tsave0.5_T500_n200/data.pt'
    dt = 1.0
    if Re == '40':
        file_dir = f'data/KF_Re{Re}_M64_tsave1_T500_n200/data.pt'
        data = torch.load(file_dir)
        data_animate = torch.load(f'data/KF_Re{Re}_M64_tsave1_T5000_n1/data.pt')
    elif Re == '500':
        file_dir = 'data/KF_Re500.0_M128_tsave1.0_T500.0_n200_ic30.0'
        train_data_raw = torch.load(file_dir)
        # data_animate = torch.load(f'data/KF_Re{Re}_M128_tsave0.5_T5000_n1/data.pt')[:,::2,::2,:]
        test_data_raw = torch.load(f'data/KF_Re{Re}_M128_tsave1.0_T2000_n10/data.pt')
        train_data = train_data_raw[:, ::2, ::2, :]
        test_data = test_data_raw[:, ::2, ::2, :]
        
        print(train_data.shape)
        print(test_data.shape)
        
        # if dt == 0.5:
        #     # Fix this later
        #     pass
        #     # train_data = train_data[::2, ...]
        #     # test_data = test_data[::2, ...]
        # if dt == 1.0:
        #     train_data = train_data[..., ::2]
        print(train_data.shape)
        print(test_data.shape)
        
    # return None
    # TODO: clarify why the subsampling is done this way
    train_data = train_data[185:, :, :, 200:]

    s = train_data.shape[1] # assuming data has shape n_traj, dim1, dim2, n_time and dim1 = dim2
    grids = []

    grids.append(np.linspace(0, 2*np.pi, s, dtype=np.float32) * trunk_scale)
    grids.append(np.linspace(2*np.pi, 0, s, dtype=np.float32) * trunk_scale) # position (0,0) of matrix is point (0,1) on plot (top left)

    if model_params['backbone'] == 'deeponet':
        x_trunk_input = torch.tensor(np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T).to(device)
    elif model_params['backbone'] == 'fno':
        x_trunk_input = None

    # lump portions of all training trajectories together for ground truth stats evaluation
    train_gt_comb = train_data.permute(0,3,1,2).reshape(-1,s*s)
    # keep trajectories separated for testing data
    test_gt_multi_traj = test_data.reshape(test_data.shape[0], test_data.shape[-1], s * s)
    
    print(train_gt_comb.shape)
    print(test_gt_multi_traj.shape)

    # Save Rollout trajectory (function requires torch)
    if os.path.exists(f'{figs_dir}/rollout_traj.pt'):
        pred_multi_traj = torch.load(f'{figs_dir}/rollout_traj.pt')
    else:
        print('Generating rollout trajectory')
        pred_multi_traj = rollout_trajectory_batched(model, 
                                    batched_initial_condition=test_gt_multi_traj[:, 0, :],
                                    figs_dir=figs_dir,
                                    s=s,
                                    n_times=test_gt_multi_traj.shape[1]-1,
                                    n_traj=test_gt_multi_traj.shape[0],
                                    trunk_input=x_trunk_input)
       
    for ii in range(test_gt_multi_traj.shape[0]):
        if torch.isnan(test_gt_multi_traj[ii]).any():
            # check which step this happens
            nan_steps = torch.isnan(test_gt_multi_traj[ii]).nonzero()
            print(f'Ground truth trajectory {ii} contains NaN values at steps: {nan_steps}')
            
    for ii in range(pred_multi_traj.shape[0]):
        if torch.isnan(pred_multi_traj[ii]).any():
            # check which step this happens
            nan_steps = torch.isnan(pred_multi_traj[ii]).nonzero()
            print(f'Rollout {ii} contains NaN values at steps: {nan_steps}')

    
    test_traj_np = test_gt_multi_traj[test_idx].detach().cpu().numpy()
    pred_traj_np = pred_multi_traj[test_idx].detach().cpu().numpy()
    
    ani_frame = 500
    ani_gt_traj = test_traj_np[:ani_frame]
    ani_pred_traj = pred_traj_np[:ani_frame]

    # Save animation of the rollout (NO torch)
    if os.path.exists(f'{figs_dir}/rollout_idx{test_idx}_{ani_frame}.gif'):
        print(f'Animation already exists: {figs_dir}/rollout_idx{test_idx}_{ani_frame}.gif')
    else:
        print('Saving animation of the rollout')
        animate_rollout(
            ani_gt_traj,
            ani_pred_traj,
            figs_dir,
            s,
            max_frames=ani_frame,
            savename=f'rollout_idx{test_idx}_{ani_frame}',
        )


    # COSINE SIMILARITY OVER TIME (currently torch)
    print('Cosine similarity over time')
    cos_steps = min(test_gt_multi_traj.shape[1], pred_multi_traj.shape[1])
    if cosine_eval_steps is not None:
        cos_steps = min(cos_steps, int(cosine_eval_steps))
    cosine_vals = None
    if cos_steps > 0:
        if os.path.exists(f'{figs_dir}/cosine_similarity.png'):
            print(f'Cosine similarity plot already exists: {figs_dir}/cosine_similarity.png')
        else:
            print('Generating cosine similarity plot')
            gt_seq = test_gt_multi_traj[:, :cos_steps, :]
            pred_seq = pred_multi_traj[:, :cos_steps, :]
            cosine_vals = plot_cos_sims(
                dt=dt,
                trajs=gt_seq,
                pred_trajs=pred_seq,
                traj_length=cos_steps,
                save_path=f'{figs_dir}/cosine_similarity.png',
            )
            print(f'Final cosine similarity: {cosine_vals[-1]:.4f}')
    else:
        print('Insufficient samples to compute cosine similarity.')

    ## SINKHORN DIVERGENCE (uses pytorch)
    print('Sinkhorn divergence')
    sinkhorn_value = None
    sinkhorn_steps = min(test_gt_multi_traj.shape[1], pred_multi_traj.shape[1])
    if sinkhorn_steps > 0:
        try:
            sinkhorn_value = sinkhorn_div(
                x=test_gt_multi_traj[:, :sinkhorn_steps, :].reshape(-1, s * s),
                y=pred_multi_traj[:, :sinkhorn_steps, :].reshape(-1, s * s),
                epsilon=0.1,
                n_iters=150,
                max_samples=512,
                p=1,
            )
            print(f'Sinkhorn divergence (epsilon=0.1): {sinkhorn_value:.6f}')
        except (ValueError, FloatingPointError) as exc:
            print(f'Sinkhorn divergence computation failed: {exc}')
            sinkhorn_value = None
    else:
        print('Insufficient samples to compute Sinkhorn divergence.')

    ## Covariance RMSE
    print('covariance RMSE')
    cov_rmse_val = None
    if sinkhorn_steps > 0:
        try:
            cov_rmse_val = covariance_rmse(
                test_gt_multi_traj[:, :sinkhorn_steps, :],
                pred_multi_traj[:, :sinkhorn_steps, :],
            )
            print(f'covRMSE: {cov_rmse_val:.6f}')
        except ValueError as exc:
            print(f'covRMSE computation failed: {exc}')
            cov_rmse_val = None
    else:
        print('Insufficient samples to compute covRMSE.')

    ## Time Correlation Metric
    print('Time correlation metric')
    tcm_val = None
    tau_gt = tau_pred = None
    if sinkhorn_steps > 1:
        try:
            tcm_val, tau_gt, tau_pred = time_correlation_metric(
                test_gt_multi_traj[:, :sinkhorn_steps, :],
                pred_multi_traj[:, :sinkhorn_steps, :],
                dt=dt,
                positive_only=True,
            )
            print(f'TCM: {tcm_val:.6f}')
        except ValueError as exc:
            print(f'TCM computation failed: {exc}')
            tcm_val = None
    else:
        print('Insufficient samples to compute TCM.')

    pred_traj_comb = pred_multi_traj.reshape(-1, s*s)
    ## FIRST TEN PCA MODES
    print('PCA MODES (method A)')
    pca_modes(w_data=train_gt_comb,w_model=pred_traj_comb,figs_dir=figs_dir,s=s,device=torch.device('cpu'))

    ## SPATIAL CORRELATION
    print('SPATIAL CORRELATION')
    spatial_corr(u_data=train_gt_comb.detach().cpu().numpy(),
        u_model=pred_traj_comb.detach().cpu().numpy(),
        figs_dir=figs_dir,
        s=s)

    ## PCA PLOT
    print('PCA PROJECTION')
    pca_traj_gt, pca_traj_pred = visualize_ellipsoid(gt_traj = train_gt_comb, 
        pred_traj = pred_traj_comb, 
        figs_dir=figs_dir, 
        Q=Q, 
        c=c,
        tag='')

    ## DISTRIBUTION COMPARISON FOR DATA
    print('DISTRIBUTION COMPARISON FOR TRAJECTORY')
    pred_traj_np = pred_traj_comb.detach().cpu().numpy()
    kl_div_traj = compare_distributions(gt_traj = train_gt_comb.detach().cpu().numpy().ravel(), 
        pred_traj = pred_traj_np.ravel(), 
        bins = 50,
        plot=True, 
        save_name=f'{figs_dir}/distribution_traj.png')


    # ## DISTRIBUTION COMPARISON FOR PCA MODES
    print('DISTRIBUTION COMPARISON FOR PCA MODES')
    pca_histogram_eval(gt_pca=pca_traj_gt, 
        pred_pca=pca_traj_pred, 
        bins=50, 
        lim=[[-250.0, 250.0], [-250.0, 250.0]], 
        save_path=f'{figs_dir}/distribution_pca.png', 
        title_gt='Ground Truth', 
        title_pred='Prediction')

    ## FOURIER SPECTRUM
    print('FOURIER SPECTRUM COMPARISON')
    print(train_gt_comb.shape)
    print(pred_traj_comb.shape)
    fourier_spectrum_2d(gt_traj=train_gt_comb,pred_traj=pred_traj_comb,s=s,figs_dir=figs_dir,device=torch.device('cpu'))

    ## V OVER TIME
    print('Energy over time')
    # n = data_animate.shape[0]
    energy_time(gt_traj=test_gt_multi_traj[test_idx],pred_traj=pred_multi_traj[test_idx],model=model,figs_dir=figs_dir)
    # return None
    return {
        "cosine_similarity": cosine_vals,
        "sinkhorn_divergence": sinkhorn_value,
        "covariance_rmse": cov_rmse_val,
        "time_correlation_metric": tcm_val,
        "tau_gt": tau_gt,
        "tau_pred": tau_pred,
    }


def main(param_path_str, Re, cosine_eval_steps=None):
    param_path = Path(param_path_str)
    output_path = param_path.parent / "results.npz"  # Same directory as params.npz

    # Load data
    data = np.load(param_path)

    # Process
    result = run_functions(data, str(param_path.parent), Re, cosine_eval_steps=cosine_eval_steps)

    # # Save result
    if result and any(v is not None for v in result.values()):
        arrays = {k: np.asarray(v) if v is not None else np.array([]) for k, v in result.items()}
        np.savez(output_path, **arrays)
        print(f"Saved results to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python eval.py /path/to/params.npz Re [cosine_eval_steps]")
        sys.exit(1)

    print(sys.argv[1])
    print(sys.argv[2])
    cosine_eval_steps = sys.argv[3] if len(sys.argv) > 3 else None

    main(sys.argv[1], sys.argv[2], cosine_eval_steps)
