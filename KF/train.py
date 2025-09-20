from calendar import c
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
from model import DeepONet

from utils import TrajectoryDataset, load_multi_traj_data
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import logging
from utils import Normalizer, InferenceWrapper
from datetime import datetime
from utils import one_step_animation, rollout_animation, pca_modes, visualize_ellipsoid, compare_distributions, pca_histogram_eval, evaluate_fourier_spectrum, spatial_corr, fourier_spectrum_2d


def ellip_vol(model):
    d = model.V.log_diag_L.numel()
    c_val = model.c ** 2
    
    # Compute det(Q)^(-1/2)
    log_det_Q = 2 * torch.sum(model.V.log_diag_L)
    det_factor = torch.exp(- 1/2 * log_det_Q)

    # Final volume
    if model.trainable_c:
        vol = (c_val**(d/2)) * det_factor
    else:
        vol = det_factor
    return vol

# choose whether to use GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
np.random.seed(0)


def train(params):
    epochs = params['epochs']
    n_save_epochs = 50
    bsize = params['bsize']
    lam_reg_vol = params['lam_reg_vol']
    project = params['project']
    tag = params['tag']
    model_dir = params['save_dir']
    trunk_scale = params['trunk_scale']
    lr = params['lr']
    warm_start = params['warm_start']
    Re = params['Re']
    scheduler = params['scheduler'] if 'scheduler' in params else False
    dt = params['dt']

    model_folder = model_dir
    figs_folder = figs_dir = os.path.join(model_dir, 'eval_results')

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    if not os.path.exists(figs_folder):
        os.makedirs(figs_folder)

    # file_dir = 'Data/KS_data_batched_l100.53_grid512_M8_T200.0_dt0.005_dt_sample0.2_amp20.0/data.npz'
    file_dir = f'data/KF_Re{Re}_M128_tsave0.5_T500_n200/data.pt'
    # file_dir = f'data/KF_Re{Re}_M64_tsave1_T500_n200/data.pt'
    data = torch.load(file_dir)
    if dt == 0.5:
        data = data[::2,...]
    if dt == 1.0:
        data = data[...,::2]
    print(data.shape)

    train_dataset, val_dataset = load_multi_traj_data(data, trunk_scale)

    print(train_dataset.branch_inputs.shape)
    print(val_dataset.branch_inputs.shape)

    # Normalization of training data
    normalizer = Normalizer(eps=params.get('norm_eps', 1e-6))
    if params.get('normalize', False):
        normalizer.fit(train_dataset.branch_inputs)
        # train (use TRAIN mu/sigma)
        train_dataset.branch_inputs = normalizer.norm(train_dataset.branch_inputs)
        train_dataset.targets       = normalizer.norm(train_dataset.targets)
        # val (use TRAIN mu/sigma)
        val_dataset.branch_inputs = normalizer.norm(val_dataset.branch_inputs)
        val_dataset.targets       = normalizer.norm(val_dataset.targets)
    else:
        # identity normalizer
        normalizer.fit(train_dataset.branch_inputs)  # still store stats for convenience

    
    train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=bsize, shuffle=False)
    logging.info(f"Created DataLoaders with {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    # Model Optimizer Initialization
    m = s = data.shape[1]  # Assuming u_batch is of shape (num_traj, dim1, dim2, n_time)
    n = 2

    model_params = {
        'm': m,
        'n': n,
        'trainable_c': params['trainable_c'],
        'c0': params['c_init'],
        'project': params['project'],
        'diag_Q': params['diag_Q'],
        'branch_conv_channels': params['branch_conv_channels'],
        'branch_fc_dims': params['branch_fc_dims'],
        'trunk_hidden_dims': params['trunk_hidden_dims'],
        'output_dim': params['output_dim'],
        'dt': params['dt'],
        'discrete_proj': params['discrete_proj'],
        'circular_padding': params['circular_padding'],
        'activation': params['activation'],
        'trunk_last_act': params['trunk_last_act'],
    }
    # save model_params dictionary in the model location, perhaps as an npz
    np.savez(f"./{model_folder}/model_params.npz", **model_params)

    model = DeepONet(model_params).to(device)
    # Adding weight_decay and lr sceduler
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    
    # --- scheduler setup (after optimizer is created) ---
    sched_type      = params.get('sched', 'cosine')
    warmup_epochs   = int(params.get('warmup_epochs', 0))
    min_lr          = float(params.get('min_lr', 1e-5))
    epochs          = params['epochs']
    lr              = params['lr']

    def build_main_scheduler():
        if sched_type == 'cosine':
            # Smooth, monotonic decay from lr -> min_lr over (epochs - warmup_epochs)
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=min_lr
            )
        elif sched_type == 'step':
            # Drop LR by gamma every step_size epochs
            step_size = int(params.get('step_size', 2500))
            gamma     = float(params.get('gamma', 0.5))
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif sched_type == 'multistep':
            # Two drops at 50% and 75% of training
            gamma = float(params.get('gamma', 0.5))
            milestones = [int(0.5*epochs), int(0.75*epochs)]
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        elif sched_type == 'exp':
            # Exponential LR decay per epoch: lr_t = lr * gamma^t
            gamma = float(params.get('gamma', 0.9995))  # ~0.9995 for long runs
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        elif sched_type == 'plateau':
            # Reduce LR when val loss plateaus. Will call with val loss.
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, min_lr=min_lr, verbose=False
            )
        else:
            return None
        
    main_sched = build_main_scheduler()

    # Optional linear warmup for the first warmup_epochs (only for scheds that don't need val loss)
    # if warmup_epochs > 0 and sched_type in {'cosine','step','multistep','exp'}:
    #     warmup = torch.optim.lr_scheduler.LambdaLR(
    #         optimizer, lr_lambda=lambda e: min(1.0, (e+1)/max(1, warmup_epochs))
    #     )
    #     scheduler = torch.optim.lr_scheduler.SequentialLR(
    #         optimizer, schedulers=[warmup, main_sched], milestones=[warmup_epochs]
    #     )
    # else:
    #     scheduler = main_sched
    # Currently disabled above
    
    scheduler = main_sched

    is_plateau = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    
    print(f'Training with Learning rate: {lr}')
    num_params = sum(v.numel() for v in model.parameters() if v.requires_grad)
    logging.info(f'model params: {num_params}')

    if warm_start:
        logging.info('removing projection layer after initialization')
        project = False
        model.project = False

    loss_func = torch.nn.MSELoss()
    tic = time.time()

    best_loss = float('inf')
    train_losses = []
    val_losses = []
    dynamic_losses = []
    reg_losses = []
    projection_percentages = []

    # --- Main Training Loop ---
    for epoch in tqdm(range(epochs + 1)):
        model.train()
        epoch_train_loss = 0
        epoch_dynamic_loss = 0
        epoch_reg_loss = 0
        epoch_active_projection_percentage = 0.0

        if warm_start:
            if epoch == 10000:
                logging.info('Adding projection layer')
                project = True
                model.project = True
        
        # Iterate over batches from the DataLoader
        batch_idx = 0
        for x_batch, y_batch in train_loader:
            # The dataloader gives us a tuple for x_batch
            branch_batch, trunk_batch = x_batch
            
            # Move batch to the correct device
            branch_batch = branch_batch.to(device)
            trunk_batch = trunk_batch.to(device)
            # Transpose the trunk input
            
            trunk_input = trunk_batch[0]
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            
            # The model expects a tuple of (branch_input, trunk_input)
            u_pred = model((branch_batch, trunk_input))

            if project and model_params['discrete_proj']:
                epoch_active_projection_percentage += model.active_projection_percentage

            dynamic_loss = loss_func(u_pred, y_batch)

            # Calculate regularization loss if projection is enabled
            if project:
                vol = ellip_vol(model)
                reg_loss = lam_reg_vol * vol.squeeze()
            else:
                reg_loss = torch.tensor(0.0, device=device)
            
            loss = dynamic_loss + reg_loss
            loss.backward()
            # Optional gradient clipping
            if params.get('clip_grad', 0.0) and params['clip_grad'] > 0.0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params['clip_grad'])
                if epoch % n_save_epochs == 0 and batch_idx == 0:  # log once per epoch
                    logging.info(f"Grad norm before clipping: {total_norm:.3f}")

                batch_idx += 1

            # if epoch % n_save_epochs == 0:
            #     for name, param in model.named_parameters():
            #         if param.grad is not None:
            #             grad_norm = param.grad.data.norm(2).item()
            #             print(f'Gradient norm for {name}: {grad_norm:.6f}')

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1000.0)
            optimizer.step()
            epoch_train_loss += loss.item()
            epoch_dynamic_loss += dynamic_loss.item()
            epoch_reg_loss += reg_loss.item()


        # Scheulder update after each epoch
        # --- end of eval block ---
        if scheduler is not None and not is_plateau:
            scheduler.step()


        # --- Evaluation, Logging, and Checkpointing ---
        if epoch % n_save_epochs == 0:
            model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    branch_val, trunk_val = x_val
                    branch_val, trunk_val = branch_val.to(device), trunk_val.to(device)
                    trunk_val_input = trunk_val[0]
                    y_val = y_val.to(device)

                    u_val_pred = model((branch_val, trunk_val_input))
                    epoch_val_loss += loss_func(u_val_pred, y_val).item()

            avg_train_loss = epoch_train_loss / len(train_loader)
            avg_val_loss = epoch_val_loss / len(val_loader)
            avg_dynamic_loss = epoch_dynamic_loss / len(train_loader)
            avg_reg_loss = epoch_reg_loss / len(train_loader)
            avg_projection_percentage = epoch_active_projection_percentage / len(train_loader)

            if scheduler is not None and is_plateau:
                scheduler.step(avg_val_loss)  # step with metric only when you have it

            # (optional) log current LR
            lr_now = optimizer.param_groups[0]['lr']
            logging.info(f"Epoch {epoch}: LR = {lr_now:.3e}")

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            dynamic_losses.append(avg_dynamic_loss)
            reg_losses.append(avg_reg_loss)
            projection_percentages.append(avg_projection_percentage)

            plt.figure(figsize=(6, 4))
            plot_x = np.linspace(0, epoch, int(epoch / n_save_epochs + 1))

            # Create figure and first axis
            fig, ax1 = plt.subplots(figsize=(6, 4))

            # Primary y-axis plots
            ax1.plot(plot_x, train_losses, label='Training Loss')
            ax1.plot(plot_x, dynamic_losses, label='Dynamic Loss')
            ax1.plot(plot_x, val_losses, label='Val Loss')

            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Loss (log scaled)')
            ax1.set_yscale('log')
            ax1.grid(True)

            # Secondary y-axis (right side)
            ax2 = ax1.twinx()
            ax2.plot(plot_x, reg_losses, 'r--', label='Reg Loss')
            ax2.set_ylabel('Regularization Loss (log scaled)', color='r')
            ax2.set_yscale('log')
            ax2.tick_params(axis='y', labelcolor='r')

            # Combine legends from both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

            plt.title('Loss over Time')
            plt.tight_layout()
            plt.savefig(f"{figs_folder}/loss_iter.png")
            plt.close('all')

            total_time = time.time() - tic
    
            plt.figure(figsize=(6, 4))
            plt.plot(plot_x, projection_percentages, label='Active Projection %', color='green')
            plt.xlabel('Iteration')
            plt.ylabel('Percentage (%)')
            plt.title('Percentage of Batch with Active Projection')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{figs_folder}/projection_percentage_iter.png")
            plt.close('all')
            
            log_string = (f"Epoch: {epoch}/{epochs} | Train Loss: {avg_train_loss:.3e} | Dynamic Loss: {avg_dynamic_loss:.3e} | Regularization Loss: {avg_reg_loss:.3e} | Val Loss: {avg_val_loss:.3e}")
            if project and model_params['discrete_proj']:
                log_string += f" | Active Proj %: {avg_projection_percentage:.2f}"
                
            log_string += f" | Time: {total_time:.2f}s"
            logging.info(log_string)

            # Save the model if it has the best validation loss so far
            if avg_val_loss < best_loss:
                logging.info(f"New best model found at epoch {epoch} with validation loss {avg_val_loss:.3e}. Saving...")
                torch.save(model.state_dict(), f"./{model_folder}/model_epoch_best.pt")
                best_loss = avg_val_loss
                best_ind = epoch
            
            tic = time.time()

    # save model_params dictionary in the model location, perhaps as an npz
    np.savez(f"./{model_folder}/model_params.npz", **model_params)

    # after you already save model_params.npz
    np.savez(f"./{model_folder}/norm_stats.npz", mu=normalizer.mu, sigma=normalizer.sigma)

    # # save model_params dictionary in the model location, perhaps as an npz
    # np.savez(f"./{model_folder}/model_params.npz", **model_params)
    
    model.load_state_dict(torch.load(f'{model_folder}/model_epoch_best.pt',map_location=device))
    model.eval()

    eval_model = InferenceWrapper(
        base_model=model,
        normalizer=normalizer,                  # knows mu/sigma
        residual=params.get('residual', False)
    )


    ## GET MODEL PARAMETERS
    if model.project:
        Q = torch.diag(model.V._construct_Q()).detach().cpu().numpy()
        c = model.c.detach().cpu().numpy()
    else:
        Q = None
        c = 30.0

    ### LOAD DATA
    print('LOADING TEST DATA')
    file_dir = f'data/KF_Re{Re}_M128_tsave1_T5000_n1/data.pt'
    data = torch.load(file_dir)
    s = data.shape[1] # assuming data has shape n_traj, dim1, dim2, n_time and dim1 = dim2
    grids = []
    grids.append(np.linspace(0, 2*np.pi, s, dtype=np.float32) * trunk_scale)
    grids.append(np.linspace(2*np.pi, 0, s, dtype=np.float32) * trunk_scale) # position (0,0) of matrix is point (0,1) on plot (top left)

    x_trunk_input = torch.tensor(np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T).to(device)

    gt_traj = data.permute(0,3,1,2).reshape(-1,s*s).to(device)
    print(gt_traj.shape)

    ## ONE STEP COMPARISON W GROUND TRUTH
    print('ONE STEP COMPARISON')
    one_step_animation(model=eval_model,
        x_val = (gt_traj[:-1,...],x_trunk_input),
        y_val = gt_traj[1:,...],
        figs_dir=figs_dir,
        s=s)

    ## ROLLOUT COMPARISON W GROUND TRUTH
    pred_traj = rollout_animation(model=eval_model,
        x_val = (gt_traj[:-1,...],x_trunk_input),
        y_val = gt_traj[1:,...],
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
        lim=[[-200.0, 200.0], [-200.0, 200.0]], 
        save_path=f'{figs_dir}/distribution_pca.png', 
        title_gt='Ground Truth', 
        title_pred='Prediction')

    ## FOURIER SPECTRUM
    print('FOURIER SPECTRUM COMPARISON')
    print(gt_traj.shape)
    print(pred_traj.shape)
    fourier_spectrum_2d(gt_traj=gt_traj,pred_traj=pred_traj,s=s,figs_dir=figs_dir,device=device)
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, help='specify number of epochs', default=10000)
    parser.add_argument('--bsize', type=int, help='specify batch size', default=512)
    parser.add_argument('--lam_reg_vol', type=float, help='specify regularization lambda', default=1.0)
    parser.add_argument('--project', action='store_true', help='True for including projection layer', default=False)
    parser.add_argument('--tag', type=str, help='tag for file names', default='')
    parser.add_argument('--c_init', type=float, help='set initial c', default=1.0)
    parser.add_argument('--trainable_c', action='store_true', help='specify whether c is trainable')
    parser.add_argument('--trunk_scale', type=float, help='scale factor for trunk net input', default=1.0)
    parser.add_argument('--diag_Q', action='store_true', help='True for including diagonal Q')
    parser.add_argument('--dt', type=float, help='time step between two consecutive states in the trajectory', default=1.0)
    parser.add_argument('--discrete_proj', action='store_true', help='True for using discrete projection')
    parser.add_argument('--lr', type=float, help='learning rate', default=2e-5)
    parser.add_argument('--warm_start', action='store_true', help='True for adding the projection layer after training')
    parser.add_argument('--Re', help='Reynolds number of training data',default=40)
    parser.add_argument('--sched', choices=['none','cosine','step','multistep','exp','plateau'],
                    default='none', help='LR scheduler type')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Linear warmup epochs before main scheduler')
    parser.add_argument('--step_size', type=int, default=2500,
                        help='Step size for StepLR (in epochs)')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Decay factor for Step/MultiStep/Exponential')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='Minimum LR for cosine or floor for plateau')
    
    parser.add_argument('--normalize', action='store_true',
                    help='Z-score normalize branch inputs and targets using training stats')
    parser.add_argument('--norm_eps', type=float, default=1e-4,
                        help='Epsilon for std when normalizing')
    
    parser.add_argument('--clip_grad', type=float, default=0.0,
                    help='If >0, clip grad-norm to this value')



    # Model parameters
    parser.add_argument('--output_dim', type=int, default=128,
                        help='Output dimension for both branch and trunk nets.')
    
    parser.add_argument('--branch_conv_channels', type=int, nargs='*', default=[32, 64, 128],
                        help='List of output channels for branch conv layers.')

    parser.add_argument('--branch_fc_dims', type=int, nargs='+', default=[128],
                        help='List of hidden layer dimensions for branch FC net.')

    parser.add_argument('--trunk_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='List of hidden layer dimensions for trunk net.')
    
    parser.add_argument('--trunk_last_act', action='store_true',
                        help='If True, use activation on last layer of trunk net.')
    
    parser.add_argument('--activation', type=str, choices=['ReLU', 'SiLU'], default='ReLU', help='Activation function to use in the network (default: ReLU)')
    parser.add_argument('--circular_padding', action='store_true', help='True for using circular padding in conv layers')

    args = parser.parse_args()

    params = vars(args)

    reg_name = ''
    if params['trainable_c']:
        reg_name += 'cTrain'
    if params['project']:
        reg_name += '_proj'
        reg_name += f'_LamRegVol{args.lam_reg_vol}'
        reg_name += f'_C0{args.c_init}'
    if params['diag_Q']:
        reg_name += '_diagQ'
    if params['discrete_proj']:
        reg_name += 'discreteProj'
    if params['warm_start']:
        reg_name += 'warmStart'
    # before building save_dir / save_name, fix reg_name
    if params['sched'] and params['sched'] != 'none':
        reg_name += f"sched_{params['sched']}"
    if args.activation != 'ReLU':
        reg_name += f'_{args.activation}'
    if params['circular_padding']:
        reg_name += '_circPad'

    print(args.branch_conv_channels)
        
    # Set up directory for saving models and plots
    now = datetime.now()
    save_time_str = now.strftime("%m%d_%H")
    save_dir = 'Trained_Models/' + save_time_str
    save_name = f'E{args.epochs}_Re{args.Re}_TS{args.trunk_scale}_branchConv{len(args.branch_conv_channels)}_trunkHidden{len(args.trunk_hidden_dims)}_dt{args.dt}_lr{args.lr}_bsize{args.bsize}_{reg_name}_{args.tag}'
    save_dir = os.path.join(save_dir, save_name)
    params['save_dir'] = save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # # logging.basicConfig(filename=save_dir + '/' + f"loss_info_{save_name}.log", level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
    logging.basicConfig(filename=save_dir + '/' + f"loss_info.log", level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
    
    model = train(params)
    

# print(f"Training complete. Best validation loss: {best_loss:.3e} at epoch {best_ind}.")
# model.eval()
# val_onestep_visual(model, data, device, figs_dir=figs_folder)