from calendar import c
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import model
import os
from utils import TrajectoryDataset, load_multi_traj_data
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import logging
from utils import run_model_visualization

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

device = torch.device('cuda')
torch.manual_seed(0)
np.random.seed(0)


def train(params):
    epochs = params['epochs']
    n_save_epochs = 50
    bsize = params['bsize']
    lam_reg_vol = params['lam_reg_vol']
    project = params['project']
    tag = params['tag']
    save_name = params['save_name']
    trunk_scale = params['trunk_scale']

    c0 = params['c_init']
    trainable_c = params['trainable_c']

    model_folder = f'models_{save_name}'
    figs_folder = f'figs_{save_name}'

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    if not os.path.exists(figs_folder):
        os.makedirs(figs_folder)

    file_dir = 'Data/KS_data_batched_l100.53_grid512_M8_T500.0_dt0.01_amp5.0/data.npz'
    data = np.load(file_dir,allow_pickle=True)

    train_dataset, val_dataset = load_multi_traj_data(data,trunk_scale)

    train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=bsize, shuffle=False)
    logging.info(f"Created DataLoaders with {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    # Model Optimizer Initialization
    m = s = data['u_batch'].shape[2]  # Assuming u_batch is of shape (num_traj, traj_length, traj_dim)
    n = 1
    import model
    model = model.DeepONet(m,n,project = project,trainable_c = trainable_c,c0=c0).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)
    num_params = sum(v.numel() for v in model.parameters() if v.requires_grad)
    logging.info(f'model params: {num_params}')

    loss_func = torch.nn.MSELoss()
    tic = time.time()

    best_loss = float('inf')
    train_losses = []
    val_losses = []
    dynamic_losses = []
    reg_losses = []

    # --- Main Training Loop ---
    for epoch in tqdm(range(epochs + 1)):
        model.train()
        epoch_train_loss = 0
        epoch_dynamic_loss = 0
        epoch_reg_loss = 0
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
            dynamic_loss = loss_func(u_pred, y_batch)

            # Calculate regularization loss if projection is enabled
            if project:
                vol = ellip_vol(model)
                reg_loss = lam_reg_vol * vol.squeeze()
            else:
                reg_loss = torch.tensor(0.0, device=device)
            
            loss = dynamic_loss + reg_loss
            loss.backward()
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
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            dynamic_losses.append(avg_dynamic_loss)
            reg_losses.append(avg_reg_loss)

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
            plt.close()

            total_time = time.time() - tic

            logging.info(f"Epoch: {epoch}/{epochs} | Train Loss: {avg_train_loss:.3e} | Dynamic Loss: {avg_dynamic_loss:.3e} | Regularization Loss: {avg_reg_loss:.3e} | Val Loss: {avg_val_loss:.3e} | Time: {total_time:.2f}s")

            # Save the model if it has the best validation loss so far
            if avg_val_loss < best_loss:
                logging.info(f"New best model found at epoch {epoch} with validation loss {avg_val_loss:.3e}. Saving...")
                torch.save(model.state_dict(), f"./{model_folder}/model_epoch_best.pt")
                best_loss = avg_val_loss
                best_ind = epoch
            
            tic = time.time()

    model.load_state_dict(torch.load(f"./{model_folder}/model_epoch_best.pt")) 
    model.eval()
    x_val = (val_dataset.branch_inputs.to(device), val_dataset.trunk_input.to(device))
    y_val = val_dataset.targets.to(device)     
    run_model_visualization(model,x_val,y_val,s,device,figs_dir = figs_folder)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,help='specify number of epochs', default=10000)
    parser.add_argument('--bsize',type=int,help='specify batch size',default=2048)
    parser.add_argument('--lam_reg_vol',type=float,help='specify regularization lambda',default=1.0)
    parser.add_argument('--project',type=bool,help='True for including projection layer',default=False)
    parser.add_argument('--tag',type=str,help='tag for file names',default='')
    parser.add_argument('--c_init', type=float, help='set initial c', default=1.0)
    parser.add_argument('--trainable_c',type=bool,help='specify whether c is trainable',default=True)
    parser.add_argument('--trunk_scale',type=float,help='scale factor for trunk net input',default=1.0)

    args = parser.parse_args()

    params = {
    'epochs': args.epochs,
    'bsize': args.bsize,
    'lam_reg_vol': args.lam_reg_vol,
    'project': args.project,
    'tag': args.tag,
    'c_init': args.c_init,
    'trainable_c': args.trainable_c,
    'trunk_scale': args.trunk_scale
    }

    reg_name = ''
    if params['trainable_c']:
        reg_name+='cTrain'
    if params['project']:
        reg_name+='_proj'
        reg_name+=f'_LamRegVol{args.lam_reg_vol}'
        reg_name+=f'_C0{args.c_init}'

    save_name = f'E{args.epochs}_TS{args.trunk_scale}_{reg_name}_{args.tag}'

    params['save_name'] = save_name

    logging.basicConfig(filename=f"loss_info_{save_name}.log", level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

    model = train(params)
