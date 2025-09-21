import torch
from utils import one_step_animation, rollout_animation, pca_modes, visualize_ellipsoid, compare_distributions, pca_histogram_eval, evaluate_fourier_spectrum, spatial_corr, fourier_spectrum_2d,energy_time
from model import DeepONet
import sys
from pathlib import Path
import os
import numpy as np



def run_functions(params,param_path_parent,Re):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trunk_scale = 1
    m = 64*2
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
        'trunk_last_act': params['trunk_last_act']
    }

    model = DeepONet(model_params).to(device)
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
    file_dir = f'data/KF_Re{Re}_M128_tsave0.5_T500_n200/data.pt'
    data = torch.load(file_dir)[185:,:,:,200:]
    print(data.shape)
    s = data.shape[1] # assuming data has shape n_traj, dim1, dim2, n_time and dim1 = dim2
    grids = []

    data_animate = torch.load(f'data/KF_Re{Re}_M128_tsave0.5_T5000_n1/data.pt')[...,::2] # assuming dt = 1.0
    data_animate = data_animate[...,:500].permute(0,3,1,2).reshape(-1,s*s).to(device)
    grids.append(np.linspace(0, 2*np.pi, s, dtype=np.float32) * trunk_scale)
    grids.append(np.linspace(2*np.pi, 0, s, dtype=np.float32) * trunk_scale) # position (0,0) of matrix is point (0,1) on plot (top left)

    x_trunk_input = torch.tensor(np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T).to(device)

    gt_traj = data.permute(0,3,1,2).reshape(-1,s*s).to(device)
    print(gt_traj.shape)

    ## ONE STEP COMPARISON W GROUND TRUTH
    print('ONE STEP COMPARISON')
    one_step_animation(model=model,
        x_val = (data_animate[:-1,...],x_trunk_input),
        y_val = data_animate[1:,...],
        figs_dir=figs_dir,
        s=s)

    ## ROLLOUT COMPARISON W GROUND TRUTH
    pred_traj = rollout_animation(model=model,
        x_val = (data_animate[:-1,...],x_trunk_input),
        y_val = data_animate[1:,...],
        figs_dir=figs_dir,
        s=s)
    pred_traj = pred_traj.to(device)

    ## FIRST TEN PCA MODES
    print('PCA MODES (method A)')
    pca_modes(w_data=gt_traj,w_model=pred_traj,figs_dir=figs_dir,s=s,device=device)

    ## SPATIAL CORRELATION
    print('SPATIAL CORRELATION')
    spatial_corr(u_data=gt_traj.detach().cpu().numpy(),
        u_model=pred_traj.detach().cpu().numpy(),
        figs_dir=figs_dir,
        s=s)

    ## PCA PLOT
    print('PCA PROJECTION')
    pca_traj_gt, pca_traj_pred = visualize_ellipsoid(gt_traj = gt_traj, 
        pred_traj = pred_traj, 
        figs_dir=figs_dir, 
        Q=Q, 
        c=c,
        tag='')

    ## DISTRIBUTION COMPARISON FOR DATA
    print('DISTRIBUTION COMPARISON FOR TRAJECTORY')
    pred_traj_np = pred_traj.detach().cpu().numpy()
    kl_div_traj = compare_distributions(gt_traj = gt_traj.detach().cpu().numpy().ravel(), 
        pred_traj = pred_traj_np.ravel(), 
        bins = 50,
        plot=True, 
        save_name=f'{figs_dir}/distribution_traj.png')


    # ## DISTRIBUTION COMPARISON FOR PCA MODES
    print('DISTRIBUTION COMPARISON FOR PCA MODES')
    pca_histogram_eval(gt_pca=pca_traj_gt, 
        pred_pca=pca_traj_pred, 
        bins=50, 
        lim=[[-500.0, 500.0], [-500.0, 500.0]], 
        save_path=f'{figs_dir}/distribution_pca.png', 
        title_gt='Ground Truth', 
        title_pred='Prediction')

    ## FOURIER SPECTRUM
    print('FOURIER SPECTRUM COMPARISON')
    print(gt_traj.shape)
    print(pred_traj.shape)
    fourier_spectrum_2d(gt_traj=gt_traj,pred_traj=pred_traj,s=s,figs_dir=figs_dir,device=device)

    ## V OVER TIME
    print('Energy over time')
    n = data_animate.shape[0]
    energy_time(gt_traj=gt_traj[100:5100],pred_traj=pred_traj[100:5100],model=model,figs_dir=figs_dir)



def main(param_path_str,Re):
    param_path = Path(param_path_str)
    output_path = param_path.parent / "results.npz"  # Same directory as params.npz

    # Load data
    data = np.load(param_path)

    # Process
    result = run_functions(data,str(param_path.parent),Re)

    # # Save result
    # np.savez(output_path, **result)
    # print(f"Saved results to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_params.py /path/to/params.npz")
        sys.exit(1)

    print(sys.argv[1])
    print(sys.argv[2])

    main(sys.argv[1],sys.argv[2])