#!/usr/bin/env python3
# eval.py

import os, argparse, logging
import numpy as np
import torch, torch.nn as nn
import matplotlib.pyplot as plt
from model import ProjectedMLP
from utils import TrajectoryTensorDataset, gen_real_multi_traj, gen_multi_traj_scipy

# set matplotlib fontsize
plt.rcParams['font.size'] = 24
# use latex
plt.rcParams['text.usetex'] = True

class OneStepFromSubtraj(torch.utils.data.Dataset):
    def __init__(self, trajectories, stride=1):
        self.base = TrajectoryTensorDataset(trajectories, subtraj_length=2, stride=stride)
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        subtraj = self.base[idx]
        if isinstance(subtraj, torch.Tensor): subtraj = subtraj.numpy()
        return torch.from_numpy(subtraj[0].astype(np.float32)), torch.from_numpy(subtraj[1].astype(np.float32))

SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0
DT = 0.05 # Timestep used for generating trajectories, needed for model's flow prediction
# --- NEW: Quiver Plot Function ---
@torch.no_grad()
def plot_flow_field(model, device, save_path, z_slice=27):
    """
    Plots the 2D flow field on the x-y plane, comparing the ground truth
    with the model's prediction. All vectors are normalized to unit length.
    
    Also plots the 2D cross-section of the learned ellipsoid boundary.
    """
    # 1. Create a grid of (x, y) points
    x_range = np.linspace(-35, 35, 25)
    y_range = np.linspace(-35, 35, 25)
    x_grid, y_grid = np.meshgrid(x_range, y_range)

    # 2. Calculate Ground Truth Flow (derivatives)
    u_gt = SIGMA * (y_grid - x_grid)
    v_gt = x_grid * (RHO - z_slice) - y_grid

    # 3. Calculate Model's Predicted Flow
    z_grid = np.full_like(x_grid, z_slice)
    grid_states = np.stack([x_grid, y_grid, z_grid], axis=-1).reshape(-1, 3)
    grid_states_torch = torch.from_numpy(grid_states.astype(np.float32)).to(device)
    
    # The model predicts the next state x_{t+1}
    next_states_torch = model(grid_states_torch)
    
    # The flow vector points from the current state to the next state
    flow_vectors = (next_states_torch - grid_states_torch).cpu().numpy()
    u_pred = flow_vectors[:, 0].reshape(x_grid.shape)
    v_pred = flow_vectors[:, 1].reshape(x_grid.shape)

    # 4. Normalize all flow vectors to unit length for clear direction visualization
    mag_gt = np.sqrt(u_gt**2 + v_gt**2) + 1e-8 # Add epsilon for stability
    u_gt_norm, v_gt_norm = u_gt / mag_gt, v_gt / mag_gt
    
    mag_pred = np.sqrt(u_pred**2 + v_pred**2) + 1e-8 # Add epsilon for stability
    u_pred_norm, v_pred_norm = u_pred / mag_pred, v_pred / mag_pred

    # 5. Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # 6. Plot the ellipsoid boundary if the model has one
    if model.discrete_proj:
        # Calculate the energy V(x) on the grid
        energy = model.V(grid_states_torch).cpu().numpy().reshape(x_grid.shape)
        c_squared = model.c.item()**2
        
        # Plot the contour where V(x) = c^2
        ax.contour(
            x_grid, y_grid, energy, levels=[c_squared],
            colors='red', linewidths=2.5, zorder=3
        )
        # Create a dummy artist for the legend
        ax.plot([], [], color='red', linewidth=2.5, label='Ellipsoid Boundary ($V(x)=c^2$)')


    # 7. Plot the quiver fields
    # Plot ground truth flow (light gray, underneath)
    # ax.quiver(x_grid, y_grid, u_gt_norm, v_gt_norm, color='lightgray', alpha=0.9, 
            #   angles='xy', label='Ground Truth Flow', zorder=1)
    
    # Plot model's predicted flow (blue, on top)
    ax.quiver(x_grid, y_grid, u_pred_norm, v_pred_norm, color='dodgerblue', alpha=0.9,
              angles='xy', label='Model Predicted Flow', zorder=2)
    
    ax.set_xlabel(r"$w_1$")
    ax.set_ylabel(r"$w_2$")
    ax.set_aspect('equal', adjustable='box')
    # ax.legend(loc='upper right')
    # ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(x_range.min(), x_range.max())
    ax.set_ylim(y_range.min(), y_range.max())
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    
def split_by_trajectory(X_ds, val_frac=0.1, seed=0):
    """
    Splits the dataset into training and validation sets.

    If there is more than one trajectory, the split is done by trajectory.
    If there is only one trajectory, the split is done by time steps.
    """
    if X_ds.shape[0] > 1:
        # Split by trajectory
        N = X_ds.shape[0]
        rng = np.random.default_rng(seed); perm = rng.permutation(N)
        n_val = max(1,int(N*val_frac))
        return X_ds[perm[n_val:]], X_ds[perm[:n_val]]
    else:
        # Split a single trajectory by time
        N = X_ds.shape[1] # Get the length of the trajectory
        n_val = max(1, int(N * val_frac))
        
        # Split the trajectory itself
        train_traj = X_ds[:, :-n_val, :]
        val_traj = X_ds[:, -n_val:, :]
        return train_traj, val_traj

def plot_ellipsoid(ax, model):
    if not model.discrete_proj:
        return
        
    # Extract ellipsoid parameters from the model
    c = model.c.item()
    Q = model.V._construct_Q().detach().cpu().numpy()
    x0 = model.V.x_0.detach().cpu().numpy().squeeze()

    # c = 40.0
    # Q = np.eye(3)
    # x0 = model.V.x_0.detach().cpu().numpy().squeeze()

    # Find the rotation matrix and semi-axes lengths
    U, s, rotation = np.linalg.svd(Q)
    radii = c / np.sqrt(s)

    # Generate points on a unit sphere
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    # Rotate and translate the ellipsoid
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + x0

    ax.plot_surface(x, y, z, rstride=4, cstride=4, color='r', alpha=0.1)

@torch.no_grad()
def plot_energy(ax, model, traj, label, device):
    """Plots the energy of a trajectory using the learned V function."""
    traj_torch = torch.from_numpy(traj.astype(np.float32)).to(device)
    energies = model.V(traj_torch).cpu().numpy()
    ax.plot(energies, label=label)

@torch.no_grad()
def eval_one_step(val_loader, model, device):
    loss_fn = nn.MSELoss(); val_mse,n=0.0,0
    preds, gts = [], []
    for xb,yb in val_loader:
        xb,yb=xb.to(device), yb.to(device)
        y_hat = model(xb)
        val_mse += loss_fn(y_hat, yb).item(); n+=1
        preds.append(y_hat.cpu().numpy())
        gts.append(yb.cpu().numpy())

    return val_mse/max(1,n), np.concatenate(preds), np.concatenate(gts)


@torch.no_grad()
def closed_loop_rollout(model, x0_np, steps, device):
    x=torch.from_numpy(x0_np.astype(np.float32)).unsqueeze(0).to(device)
    preds=[x.cpu().numpy()[0]]
    for _ in range(steps):
        x=model(x); preds.append(x.detach().cpu().numpy()[0])
    return np.stack(preds,0)


def plot_traj3d(xyz_gt, xyz_pred, save_path, title, model, stride=1):
    """
    Plot a single 3D Lorenz-63 trajectory from X_ds with shape (N, T, 3).
    """
    if stride > 1:
        xyz_gt = xyz_gt[::stride]
        xyz_pred = xyz_pred[::stride]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    plot_ellipsoid(ax, model)
    
    ax.plot(xyz_gt[:,0], xyz_gt[:,1], xyz_gt[:,2], label='Ground Truth')
    ax.plot(xyz_pred[:,0], xyz_pred[:,1], xyz_pred[:,2], label='Prediction')

    ax.set_xlabel(r"$w_1$")
    ax.set_ylabel(r"$w_2$")
    ax.set_zlabel(r"$w_3$")
    # ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-dir',required=True)
    p.add_argument('--val-frac',type=float,default=0.2)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--ckpt',default='model_epoch_best.pt')
    p.add_argument('--eval-steps',type=int,default=1000)
    p.add_argument('--gt-traj-num',type=int,default=1)
    p.add_argument('--gt-seed',type=int,default=123)
    args=p.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    params_npz=np.load(os.path.join(args.model_dir,'model_params.npz'),allow_pickle=True)

    data_path = str(params_npz['data_path'])
    # print(data_path)

    X_ds=np.load(data_path,allow_pickle=True)['X_ds']

    _,T,D=X_ds.shape
    
    model_cfg={k:params_npz[k].item() if params_npz[k].size==1 else params_npz[k].tolist()
               for k in params_npz.files if k in ['d','hidden_dims','activation','discrete_proj','c0','trainable_c','diag_Q']}
    
    model_cfg['activation']=nn.GELU() if params_npz['activation']=='gelu' else nn.ReLU()

    model=ProjectedMLP(model_cfg).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir,args.ckpt),map_location=device))

    _,X_val=split_by_trajectory(X_ds,args.val_frac,args.seed)
    val_loader=torch.utils.data.DataLoader(OneStepFromSubtraj(X_val),batch_size=4096,shuffle=False)
    
    results_dir = os.path.join(args.model_dir, 'eval_results')
    os.makedirs(results_dir, exist_ok=True)

    # --- (1) Validation trajectory one-step ---
    one_step_val_mse, one_step_preds, one_step_gts = eval_one_step(val_loader,model,device)
    plot_traj3d(one_step_gts, one_step_preds, os.path.join(results_dir, 'eval_one_step_val.png'), 'One-Step Validation Predictions', model)

    # --- (2) Validation trajectory rollout ---
    steps=min(args.eval_steps,T-1)
    val_traj=X_val[0]
    steps = val_traj.shape[0] - 1
    pred_val=closed_loop_rollout(model,val_traj[0],steps,device)
    rollout_val_mse=float(np.mean((pred_val-val_traj[:steps+1])**2))
    plot_traj3d(val_traj[:steps+1], pred_val, os.path.join(results_dir, 'eval_rollout_val.png'), 'Rollout on Validation Set', model)
    # print(val_traj[:, 1] - val_traj[:, 0])

    plt.figure(); [plt.plot(val_traj[:steps+1,i],label=f'gt{i}') or plt.plot(pred_val[:,i],'--',label=f'pred{i}') for i in range(min(3,D))]
    plt.legend(); plt.savefig(os.path.join(results_dir,'eval_rollout_val_traj.png')); plt.close()

    # --- (3) Random initial condition rollout ---
    X_gt=gen_multi_traj_scipy(M=args.gt_traj_num,N=steps,dt_target=0.05,seed=1)
    if isinstance(X_gt,dict) and "X_ds" in X_gt: X_gt=X_gt['X_ds']
    
    # Plot the first GT trajectory and its rollout
    gt_traj_to_plot = X_gt[0]
    pred_gt = closed_loop_rollout(model, gt_traj_to_plot[0], gt_traj_to_plot.shape[0]-1, device)
    plot_traj3d(gt_traj_to_plot, pred_gt, os.path.join(results_dir, 'eval_rollout_gt.png'), 'Rollout on Random Initial Condition', model)

    # --- (4) Energy evaluation plot ---
    if model.discrete_proj:
        print(np.linalg.eigvals(model.V._construct_Q().detach().cpu().numpy()))
        print(model.c.item())
        print(model.V.x_0.detach().cpu().numpy())
        
        fig_energy, ax_energy = plt.subplots(figsize=(10, 10))
        # plot_energy(ax_energy, model, val_traj[:steps+1], 'Ground Truth (Validation)', device)
        # plot_energy(ax_energy, model, pred_val, 'Prediction (Validation)', device)
        plot_energy(ax_energy, model, X_gt[0], 'Ground Truth ', device)
        plot_energy(ax_energy, model, pred_gt, 'Prediction ', device)
        ax_energy.axhline(y=model.c.item()**2, color='r', linestyle='--', label='c (Energy Level)')
        ax_energy.set_xlabel('Time Steps')
        ax_energy.set_ylabel('Energy (V(w(t)))')
        ax_energy.set_ylim([0, 2000.0])
        ax_energy.legend()
        ax_energy.grid(True)
        fig_energy.savefig(os.path.join(args.model_dir, 'eval_energy.png'), dpi=300)
        plt.close(fig_energy)

    # mse_list=[np.mean((closed_loop_rollout(model,tr[0],steps,device)-tr)**2) for tr in X_gt]
    # rollout_gt_mse=float(np.mean(mse_list))
    
    print("\nGenerating flow field comparison plot...")
    plot_flow_field(
        model=model,
        device=device,
        save_path=os.path.join(results_dir, 'eval_flow_field.png'),
        z_slice=27.0  # A meaningful slice at the z-level of the fixed points
    )

    print(f"One-step Val MSE: {one_step_val_mse:.6e}")
    print(f"Rollout Val MSE:  {rollout_val_mse:.6e}")
    # print(f"Rollout GT MSE:   {rollout_gt_mse:.6e}")


if __name__=="__main__":
    logging.basicConfig(level=logging.INFO); main()