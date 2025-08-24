from calendar import c
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from model import DeepONet
import os
from utils import TrajectoryDataset, load_multi_traj_data, val_onestep_visual
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import logging
from utils import run_model_visualization
from utils import visualize_ellipsoid
from utils import rollout_on_test
from datetime import datetime
from utils import visualize_ellipsoid, compare_distributions, rollout_on_test, pca_histogram_eval, evaluate_fourier_spectrum, run_model_visualization

### TRAIN FILE WITHOUT UPDATES TO LEARNING RATE - CURRENTLY WORKS FOR BLOWUP WITHOUT PROJECTION, GOOD STATS WITH PROJECTION
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

    model_folder = model_dir
    figs_folder = figs_dir = os.path.join(model_dir, 'eval_results')

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    if not os.path.exists(figs_folder):
        os.makedirs(figs_folder)

    # file_dir = 'Data/KS_data_batched_l100.53_grid512_M8_T200.0_dt0.005_dt_sample0.2_amp20.0/data.npz'
    logging.info('Using data with dt')
    file_dir = 'Data/KS_data_batched_l100.53_grid512_M8_T500.0_dt0.01_amp5.0/data.npz'
    data = np.load(file_dir, allow_pickle=True)

    train_dataset, val_dataset = load_multi_traj_data(data, trunk_scale)

    train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=bsize, shuffle=False)
    logging.info(f"Created DataLoaders with {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    # Model Optimizer Initialization
    m = s = data['u_batch'].shape[2]  # Assuming u_batch is of shape (num_traj, traj_length, traj_dim)
    n = 1

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
    }

    model = DeepONet(model_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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

    model.load_state_dict(torch.load(f'{model_folder}/model_epoch_best.pt',map_location=device))
    model.eval()

    ## GET MODEL PARAMETERS
    if model.project:
        Q = model.V._construct_Q().detach().cpu().numpy()
        c = model.c.detach().cpu().numpy() ** 2
    else:
        Q = None
        c = 30.0 ** 2

    ### LOAD DATA
    print('LOADING TEST DATA')
    trunk_scale = 0.05
    file_dir = 'Data/KS_data_test_l100.53_grid512_M1_T2000.0_dt0.005_dt_sample0.2_amp20.0.npz/data.npz'
    data = np.load(file_dir, allow_pickle=True)
    x = torch.tensor(data['x'],dtype=torch.float32).to(device)
    x = x.reshape(s,1)
    gt_traj = data['u_batch'][:,::5,:]
    print(gt_traj.shape)

    ## ONE STEP ON TEST DATA AND ROLLOUT ON RANDOM IC
    print('ONE STEP AND RANDOM IC VISUALS')
    run_model_visualization(
        model = model,
        x_test=(torch.tensor(gt_traj[0,:999,:],dtype=torch.float32).to(device),x*trunk_scale),
        y_test=torch.tensor(gt_traj[0,1:1000,:],dtype=torch.float32).to(device),
        s=s,
        device=device,
        figs_dir=figs_dir,
        figs_tag = '',
        rollout_steps_test=2000,
        rollout_steps_random=10000,
        random_seed=10,
        random_IC_mag=20.0
        )

    ## ROLLOUT TRAJECTORY AND V PLOT
    print('ROLLOUT TRAJECTORY')
    pred_traj = rollout_on_test(model, 
        data_x=x, 
        trunk_scale=trunk_scale, 
        test_traj=gt_traj, 
        device=device, 
        figs_dir=figs_dir, 
        project=model.project,
        c=c
        )
    print(pred_traj.shape)

    ## PCA PLOT
    print('PCA PROJECTION')
    pca_traj_gt, pca_traj_pred = visualize_ellipsoid(gt_traj = gt_traj[0,...], 
        test_traj = pred_traj[0,...], 
        figs_dir=figs_dir, 
        Q=Q, 
        c=c,
        tag='')

    ## DISTRIBUTION COMPARISON FOR DATA
    print('DISTRIBUTION COMPARISON FOR TRAJECTORY')
    pred_traj_np = pred_traj[0,...].detach().cpu().numpy()
    kl_div_traj = compare_distributions(gt_traj = gt_traj[0,...].ravel(), 
        pred_traj = pred_traj_np.ravel(), 
        bins = 50,
        plot=True, 
        save_name=f'{figs_dir}/distribution_traj.png')


    # ## DISTRIBUTION COMPARISON FOR PCA MODES
    print('DISTRIBUTION COMPARISON FOR PCA MODES')
    pca_histogram_eval(gt_pca=pca_traj_gt, 
        pred_pca=pca_traj_pred, 
        bins=50, 
        lim=[[-50.0, 50.0], [-50.0, 50.0]], 
        save_path=f'{figs_dir}/distribution_pca.png', 
        title_gt='Ground Truth', 
        title_pred='Prediction')

    ## FOURIER SPECTRUM
    print('FOURIER SPECTRUM COMPARISON')
    print(gt_traj.shape)
    print(pred_traj.shape)
    final_error_star = evaluate_fourier_spectrum(gt_traj = gt_traj[0,...], 
        star_traj=pred_traj_np, 
        save_path=f'{figs_dir}/fourier_spectrum.png')


    ## PLOT GT AND PREDICTED TRAJ
    aspect = 1/2*pred_traj.shape[1]/pred_traj.shape[2]
    plt.figure()
    plt.imshow(
        pred_traj[0,...].T.detach().cpu().numpy().astype(np.float32),
        # extent=[0, rollout_steps_random, 0, s],
        vmin = -5, vmax = 5,
        aspect=aspect
    )
    plt.title('Rollout from Validation Initial Condition')
    plt.colorbar()
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.savefig(f'{figs_dir}/rollout.png')
    plt.close()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, help='specify number of epochs', default=20000)
    parser.add_argument('--bsize', type=int, help='specify batch size', default=2048)
    parser.add_argument('--lam_reg_vol', type=float, help='specify regularization lambda', default=1.0)
    parser.add_argument('--project', action='store_true', help='True for including projection layer', default=False)
    parser.add_argument('--tag', type=str, help='tag for file names', default='')
    parser.add_argument('--c_init', type=float, help='set initial c', default=1.0)
    parser.add_argument('--trainable_c', action='store_true', help='specify whether c is trainable')
    parser.add_argument('--trunk_scale', type=float, help='scale factor for trunk net input', default=1.0)
    parser.add_argument('--diag_Q', action='store_true', help='True for including diagonal Q')
    parser.add_argument('--dt', type=float, help='time step between two consecutive states in the trajectory', default=0.2)
    parser.add_argument('--discrete_proj', action='store_true', help='True for using discrete projection')
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-4)
    parser.add_argument('--warm_start', action='store_true', help='True for adding the projection layer after training')

    # Model parameters
    parser.add_argument('--output_dim', type=int, default=128,
                        help='Output dimension for both branch and trunk nets.')
    
    parser.add_argument('--branch_conv_channels', type=int, nargs='*', default=[32, 64, 128],
                        help='List of output channels for branch conv layers.')

    parser.add_argument('--branch_fc_dims', type=int, nargs='+', default=[128],
                        help='List of hidden layer dimensions for branch FC net.')

    parser.add_argument('--trunk_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='List of hidden layer dimensions for trunk net.')

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

    print(args.branch_conv_channels)
        
    # Set up directory for saving models and plots
    now = datetime.now()
    save_time_str = now.strftime("%m%d_%H")
    save_dir = 'Trained_Models/' + save_time_str
    save_name = f'E{args.epochs}_TS{args.trunk_scale}_branchConv{len(args.branch_conv_channels)}_trunkHidden{len(args.trunk_hidden_dims)}_dt{args.dt}_{reg_name}_{args.tag}'
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