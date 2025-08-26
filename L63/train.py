#!/usr/bin/env python3
# train_projected_mlp.py

import os, time, argparse, logging
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import ProjectedMLP
from utils import TrajectoryTensorDataset


def ellip_vol(model: ProjectedMLP):
    
    d = model.V.log_diag_L.numel()
    c_val = model.c ** 2
    log_det_Q = 2.0 * torch.sum(model.V.log_diag_L)
    det_factor = torch.exp(-0.5 * log_det_Q)
    return (c_val ** (d / 2.0)) * det_factor if model.trainable_c else det_factor


class OneStepFromSubtraj(Dataset):
    """Wraps TrajectoryTensorDataset with subtraj_length=2 to emit (x_t, x_{t+1}) pairs, raw units."""
    def __init__(self, trajectories: np.ndarray, stride: int = 1):
        self.base = TrajectoryTensorDataset(trajectories, subtraj_length=2, stride=stride)

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        subtraj = self.base[idx]
        if isinstance(subtraj, torch.Tensor): subtraj = subtraj.numpy()
        x_in, x_out = subtraj[0], subtraj[1]
        return torch.from_numpy(x_in.astype(np.float32)), torch.from_numpy(x_out.astype(np.float32))


def split_by_trajectory(X_ds: np.ndarray, val_frac: float = 0.1):
    """
    Splits the dataset into training and validation sets.

    If there is more than one trajectory, the split is done by trajectory.
    If there is only one trajectory, the split is done by time steps.
    """
    if X_ds.shape[0] > 1:
        # Split by trajectory
        N = X_ds.shape[0]
        n_val = max(1, int(N * val_frac))
        return X_ds[n_val:], X_ds[:n_val]
    else:
        # Split a single trajectory by time
        N = X_ds.shape[1] # Get the length of the trajectory
        n_val = max(1, int(N * val_frac))
        
        # Split the trajectory itself
        train_traj = X_ds[:, :-n_val, :]
        val_traj = X_ds[:, -n_val:, :]
        return train_traj, val_traj

def plot_losses(plot_x, train_losses, val_losses, dyn_losses, reg_losses, out_path):
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(plot_x, train_losses, label='Training Loss')
    ax1.plot(plot_x, dyn_losses, label='Dynamic Loss')
    ax1.plot(plot_x, val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss (log)')
    ax1.set_yscale('log'); ax1.grid(True)
    ax2 = ax1.twinx()
    ax2.plot(plot_x, reg_losses, 'r--', label='Reg Loss')
    ax2.set_ylabel('Reg Loss (log)', color='r'); ax2.set_yscale('log')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc='upper right')
    plt.tight_layout(); plt.savefig(out_path); plt.close()


def plot_proj_vs_qgrad(plot_x, proj_perc, q_grad_norms, out_path):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Active Projection (%)', color='tab:green')
    ax1.plot(plot_x, proj_perc, color='tab:green')
    ax1.tick_params(axis='y', labelcolor='tab:green'); ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Avg Q Grad Norm (log)', color='tab:purple'); ax2.set_yscale('log')
    ax2.plot(plot_x, q_grad_norms, color='tab:purple', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:purple')
    plt.tight_layout(); plt.savefig(out_path); plt.close()


def train(params):
    torch.manual_seed(params['seed']); np.random.seed(params['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")

    # === Load dataset ===
    data_path = params['data_path']
    X_ds = np.load(data_path, allow_pickle=True)['X_ds'] if data_path.endswith('.npz') else np.load(data_path, allow_pickle=True)
    _, _, D = X_ds.shape
    
    # === Split ===
    X_train, X_val = split_by_trajectory(X_ds, val_frac=params['val_frac'])
    train_set = OneStepFromSubtraj(X_train, stride=1)
    val_set   = OneStepFromSubtraj(X_val,   stride=1)
    train_loader = DataLoader(train_set, batch_size=params['bsize'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=params['bsize'], shuffle=False, num_workers=2)

    x_mean = X_train.mean(axis=1, keepdims=True)
    # print(x_mean[0][0])

    # === Model ===
    model_params = {
        'd': D,
        'hidden_dims': params['hidden_dims'],
        'activation': nn.GELU() if params['activation']=='gelu' else nn.ReLU(),
        'discrete_proj': params['discrete_proj'],
        'c0': params['c_init'],
        'trainable_c': params['trainable_c'],
        'diag_Q': params['diag_Q'],
        'data_path': data_path,
        'x_0': x_mean[0][0],
    }

    
    save_dir, figs_dir = params['save_dir'], os.path.join(params['save_dir'], 'eval_results')
    os.makedirs(figs_dir, exist_ok=True)
    
    np.savez(os.path.join(save_dir,"model_params.npz"), **model_params)
    
    model = ProjectedMLP(model_params).to(device)
    print(f"Initial x_0 is {model.V.x_0}")

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    loss_fn = nn.MSELoss()

    best_val, tic = float('inf'), time.time()
    train_losses, val_losses, dyn_losses, reg_losses, proj_percs, q_grad_norms = [], [], [], [], [], []

    for epoch in tqdm(range(params['epochs'] + 1)):
        model.train()
        ep_train = ep_dyn = ep_reg = ep_proj = ep_qgrad = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            yhat = model(xb)
            if params['discrete_proj']: 
                ep_proj += model.active_projection_percentage

            dyn_loss = loss_fn(yhat, yb)

            reg_loss = params['lam_reg_vol'] * ellip_vol(model).squeeze() if params['discrete_proj'] else torch.tensor(0.0, device=device)

            loss = dyn_loss + reg_loss
            loss.backward()
            
            if model_params['discrete_proj']: 
                ep_qgrad += model.V.log_diag_L.grad.data.norm(2).item() if model.V.log_diag_L.grad is not None else 0.0
                
            optimizer.step()
            
            ep_train += loss.item()
            ep_dyn += dyn_loss.item()
            ep_reg += reg_loss.item()
        
        scheduler.step()
        
        if epoch % params['n_save_epochs'] == 0:
            model.eval(); ep_val = 0.0
            with torch.no_grad():
                for xb, yb in val_loader: ep_val += loss_fn(model(xb.to(device)), yb.to(device)).item()
            avg_train, avg_val = ep_train/len(train_loader), ep_val/len(val_loader)
            train_losses.append(avg_train); val_losses.append(avg_val)
            dyn_losses.append(ep_dyn/len(train_loader)); reg_losses.append(ep_reg/len(train_loader))
            proj_percs.append(ep_proj/len(train_loader)); q_grad_norms.append(ep_qgrad/len(train_loader))

            plot_x = np.linspace(0, epoch, len(train_losses))
            plot_losses(plot_x, train_losses, val_losses, dyn_losses, reg_losses, os.path.join(figs_dir,"loss_iter.png"))
            plot_proj_vs_qgrad(plot_x, proj_percs, q_grad_norms, os.path.join(figs_dir,"proj_vs_grad_norm_iter.png"))
            
            # logging the loss values
            logging.info(f"Epoch {epoch}: Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}, "
                         f"Dyn Loss: {dyn_losses[-1]:.4f}, Reg Loss: {reg_losses[-1]:.4f}, "
                         f"Proj Perc: {proj_percs[-1]:.2f}%, Q Grad Norm: {q_grad_norms[-1]:.4f}")

            # torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch}.pt"))
            if avg_val < best_val:
                best_val = avg_val
                torch.save(model.state_dict(), os.path.join(save_dir,"model_epoch_best.pt"))
            tic = time.time()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data-path', type=str, default="Data/L63_M1_N200000_dt_s0.05_dt0.001_ic50.0_intrk45/data.npz")
    parser.add_argument('--data-path', type=str, default="Data/L63_M10_N20000_dt_s0.05_dt0.001_ic50.0_intscipy/data.npz")
    parser.add_argument('--val-frac', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--n-save-epochs', type=int, default=50)
    parser.add_argument('--bsize', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[150, 150, 150, 150, 150, 150])
    parser.add_argument('--activation', type=str, choices=['relu','gelu'], default='relu')
    parser.add_argument('--discrete-proj', action='store_true')
    parser.add_argument('--diag_Q', action='store_true', default=False)
    parser.add_argument('--trainable_c', action='store_true', default=False)
    parser.add_argument('--c-init', dest='c_init', type=float, default=100.0)
    parser.add_argument('--lam-reg-vol', type=float, default=0.01)
    parser.add_argument('--tag', type=str, default='')

    args = parser.parse_args()
    params = vars(args)

    now = datetime.now().strftime("%m%d_%H")
    save_dir = os.path.join('Trained_Models', now,
                            f"E{params['epochs']}_lam_{params['lam_reg_vol']}_lr{params['lr']}_proj_{params['discrete_proj']}_layer{len(params['hidden_dims'])}_act{params['activation']}_c0{params['c_init']}_trainc_{params['trainable_c']}")
    os.makedirs(save_dir, exist_ok=True); params['save_dir'] = save_dir
    logging.basicConfig(filename=os.path.join(save_dir,"loss_info.log"),
                        level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
    _ = train(params)
