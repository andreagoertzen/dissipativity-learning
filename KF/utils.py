import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

import os
os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from tqdm import tqdm
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.special import rel_entr
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
import scipy
import scipy.io
import matplotlib.animation as animation
from tqdm import tqdm
from gstools import SRF, Gaussian
import warnings

try:
    import ot  # Python Optimal Transport
except ImportError:  # pragma: no cover - optional dependency
    ot = None

class TrajectoryDataset(Dataset):
    """
    Custom PyTorch Dataset for DeepONet trajectory data.
    
    This class takes trajectory data of shape (num_traj, traj_length, traj_dim)
    and creates input-output pairs of (u_t, u_{t+1}).
    """
    def __init__(self, u_data, x_data):
        """
        Args:
            u_data (np.array): Trajectory data with shape (num_traj, traj_length, traj_dim).
            x_data (np.array): Constant input for the Trunk network.
        """
        super().__init__()
        # The trunk input is constant for all samples, so convert it once.
        # if x_data.ndim == 1:
        #     x_data = x_data.reshape(-1, 1)

        print('x_data shape')
        print(x_data.shape)

        self.trunk_input = torch.tensor(x_data, dtype=torch.float32)
        
        branch_inputs = []
        targets = []
        
        # Iterate over each trajectory to create (u_t, u_{t+1}) pairs
        s = u_data.shape[1] # assuming data has shape n_traj, dim1, dim2, n_time and dim1 = dim2
        for traj in u_data:
            # A trajectory of length L has L-1 possible pairs
            for t in range(traj.shape[-1] - 1):
                branch_inputs.append(traj[...,t].reshape(-1,s*s))
                targets.append(traj[...,t+1].reshape(-1,s*s))
                
        # Convert the lists of individual steps into single large tensors for efficiency.
        self.branch_inputs = torch.tensor(np.array(branch_inputs), dtype=torch.float32).squeeze(1)
        self.targets = torch.tensor(np.array(targets), dtype=torch.float32).squeeze(1)
        
    def __len__(self):
        """Returns the total number of (u_t, u_{t+1}) pairs."""
        return self.branch_inputs.shape[0]
    
    def __getitem__(self, idx):
        """
        Fetches the sample at the given index.
        
        Returns:
            tuple: A tuple containing ((branch_input, trunk_input), target).
                   This format is convenient for unpacking during the training loop.
        """
        branch_in = self.branch_inputs[idx]
        target = self.targets[idx]
        
        model_input = (branch_in, self.trunk_input)
        
        return model_input, target
    
class Multistep_Dataset(Dataset):
    """
    Custom PyTorch Dataset for DeepONet trajectory data.
    
    This class takes trajectory data of shape (num_traj, traj_length, traj_dim)
    and creates input-output pairs of (u_t, u_{t+1}).
    """
    def __init__(self, u_data, x_data, stride, n_steps):
        """
        Args:
            u_data (np.array): Trajectory data with shape (num_traj, traj_length, traj_dim).
            x_data (np.array): Constant input for the Trunk network.
            stride (int): Time step between sampling multi-step training data from a single trajectory.
            n_steps (int): Number of time steps to predict ahead.
        """
        super().__init__()
        # The trunk input is constant for all samples, so convert it once.
        # if x_data.ndim == 1:
        #     x_data = x_data.reshape(-1, 1)

        print('x_data shape')
        print(x_data.shape)
        
        self.stride = stride
        self.n_steps = n_steps

        self.trunk_input = torch.tensor(x_data, dtype=torch.float32)
        self.indices = []
        self.u_data = torch.tensor(u_data, dtype=torch.float32)

        num_traj, dim_1, dim_2, traj_length = u_data.shape
        self.u_stack = torch.reshape(u_data, (num_traj, dim_1*dim_2, traj_length))
        for traj_index in range(num_traj):
            for t in range(0, traj_length - n_steps + 1, stride):
                self.indices.append((traj_index, t))
        
    def __len__(self):
        """Returns the total number of (u_t, u_{t+1}) pairs."""
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Fetches the sample at the given index.
        
        Returns:
            tuple: A tuple containing ((branch_input, trunk_input), target).
                   This format is convenient for unpacking during the training loop.
        """
        traj_index, t = self.indices[idx]
        branch_in = self.u_stack[traj_index, :, t]
        target = self.u_stack[traj_index, :, (t+1): (t+self.n_steps)]

        model_input = (branch_in, self.trunk_input)
        
        return model_input, target
    
def load_multi_traj_data(data, trunk_scale, multi_step=False, stride=1, n_steps=3):
    s = data.shape[1] # assuming data has shape n_traj, dim1, dim2, n_time and dim1 = dim2
    grids = []
    grids.append(np.linspace(0, 2*np.pi, s, dtype=np.float32) * trunk_scale)
    grids.append(np.linspace(2*np.pi, 0, s, dtype=np.float32) * trunk_scale) # position (0,0) of matrix is point (0,1) on plot (top left)

    x_trunk_input = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

    print('X_TRUNK_INPUT')
    print(x_trunk_input)

    u_all_traj = data
    # x_trunk_input = data['x']*trunk_scale
    num_traj = u_all_traj.shape[0]
    print('num traj: ')
    print(num_traj)

    # Split trajectories into training and validation sets (80/20)
    # It's important to split trajectories, not individual time-steps,
    # to prevent data leakage between train and validation.
    num_train_traj = int(0.8 * num_traj)

    u_train_traj = u_all_traj[:num_train_traj,...]
    u_val_traj = u_all_traj[num_train_traj:,...]

    if multi_step:
        train_dataset = Multistep_Dataset(u_data=u_train_traj, x_data=x_trunk_input, stride=stride, n_steps=n_steps)
        val_dataset = Multistep_Dataset(u_data=u_val_traj, x_data=x_trunk_input, stride=stride, n_steps=n_steps)
    else:
        train_dataset = TrajectoryDataset(u_data=u_train_traj, x_data=x_trunk_input)
        val_dataset = TrajectoryDataset(u_data=u_val_traj, x_data=x_trunk_input)

    return train_dataset, val_dataset


def one_step_animation(model,x_val,y_val,figs_dir,s):
    ims = []
    n_times = y_val.shape[0]
    fig,axs = plt.subplots(2)  
    axs[0].set_title('Data') 
    axs[1].set_title('Model')
    fig.tight_layout()  

    vmin = torch.min(y_val)
    vmax = torch.max(y_val)
    print(type(y_val))

    ## animation to compare to a single trajectory
    with torch.no_grad():
        for i in tqdm(range(n_times)):

            y_pred = model((x_val[0][i,...],x_val[1]))
            im = axs[0].imshow(y_val[i,:].reshape(s,s).detach().cpu().numpy(),animated = 'True',cmap='RdBu',vmin=vmin, vmax=vmax)#vmin=np.min(y_val[i,:]),vmax=np.max(y_val[i,:]))
            im2 = axs[1].imshow(y_pred.reshape(s,s).detach().cpu().numpy(),animated = 'True',cmap='RdBu',vmin=vmin,vmax=vmax)#vmin=np.min(y_val[i,:]),vmax=np.max(y_val[i,:]))

            ims.append([im,im2])


    ani = animation.ArtistAnimation(fig,ims,interval = 1e-6)
    print('saving animation')
    update_func = lambda _i, _n: progress_bar.update(1)
    with tqdm(total=n_times, desc='Saving video') as progress_bar:
        ani.save(f"{figs_dir}/one_step.gif",progress_callback=update_func)

def rollout_animation(model, x_val, y_val, figs_dir, s):
    ims = []
    n_times = 10000
    n_animate = 499 #y_val.shape[0]
    print(n_animate)
    fig,axs = plt.subplots(2)  
    axs[0].set_title('Data') 
    axs[1].set_title('Model')
    fig.tight_layout()  

    vmin = torch.min(y_val)
    vmax = torch.max(y_val)

    pred_traj = torch.zeros(n_times+1,s*s)
    # pred_traj[0,...] = x_val[0][0,...]
    x = y = range(s)
    rf = Gaussian(dim = 2, var = 1, len_scale = 10)
    srf = SRF(rf,seed = 13,generator='Fourier',period = s)
    field = srf.structured([x,y],seed=900)
    pred_traj[0,...] = torch.tensor(field).reshape(-1,s*s) 

    ## animation to compare to a single trajectory
    y_pred = model((x_val[0][0,...],x_val[1]))
    with torch.no_grad():
        for i in tqdm(range(n_times)):

            if i < n_animate:
                im = axs[0].imshow(y_val[i,:].reshape(s,s).detach().cpu().numpy(),animated = 'True',cmap='RdBu',vmin=vmin, vmax=vmax)
                im2 = axs[1].imshow(y_pred.reshape(s,s).detach().cpu().numpy(),animated = 'True',cmap='RdBu',vmin=vmin,vmax=vmax)
                ims.append([im,im2])
            pred_traj[i+1,...] = y_pred
            y_pred = model((y_pred,x_val[1]))


    ani = animation.ArtistAnimation(fig,ims,interval = 1e-6)
    print('saving animation')
    update_func = lambda _i, _n: progress_bar.update(1)
    with tqdm(total=n_animate, desc='Saving video') as progress_bar:
        ani.save(f"{figs_dir}/rollout.gif",progress_callback=update_func)#, writer = 'ffmpeg')
    torch.save(pred_traj,f'./{figs_dir}/rollout_traj.pt')
    return pred_traj

def generate_unique_ics(srf, x, y, n_traj, device, s, init_scale):
    ics = []
    base_seed = 900 # Starting seed
    for k in range(n_traj):
        # Sample a new field by changing the seed
        field = srf.structured([x, y], seed=base_seed + k*12)
        field = torch.tensor(field, dtype=torch.float32, device=device).reshape(1, s * s)
        ics.append(field)
    
    # Concatenate all unique fields and apply scale
    initial_condition_batch = torch.cat(ics, dim=0) * init_scale
    return initial_condition_batch

def rollout_trajectory_batched(model, batched_initial_condition, figs_dir, s, n_times=2000-1, n_traj=5, use_GT_init=True, init_scale=1.0, trunk_input=None):
    device = next(model.parameters()).device

    pred_traj = torch.zeros(n_traj, n_times + 1, s * s, device=device)
    if use_GT_init:
        pred_traj[:, 0, :] = batched_initial_condition
    else:
        x = y = range(s)
        rf = Gaussian(dim = 2, var = 1, len_scale = 10)
        srf = SRF(rf,seed = 13,generator='Fourier',period = s)
        initial_condition_batch = generate_unique_ics(srf, x, y, n_traj, device, s, init_scale)
        pred_traj[:, 0, :] = initial_condition_batch

    with torch.no_grad():
        for i in tqdm(range(n_times)):
            if model.backbone == 'deeponet':
                pred_traj[:, i + 1, :] = model((pred_traj[:, i, :], trunk_input))
            elif model.backbone == 'fno':
                pred_traj[:, i + 1, ...] = model((pred_traj[:, i, :], trunk_input))

    pred_traj_cpu = pred_traj.detach().cpu()
    torch.save(pred_traj_cpu, f"./{figs_dir}/rollout_traj.pt")

    return pred_traj_cpu

def animate_rollout(gt_traj, pred_traj, figs_dir, s, max_frames=500, fps=5, savename='rollout'):
    ims = []
    n_animate = max_frames - 1
    print(n_animate)
    fig,axs = plt.subplots(2)  
    axs[0].set_title('Data') 
    axs[1].set_title('Model')
    fig.tight_layout()  

    # Convert inputs to numpy arrays for Matplotlib/NumPy utilities
    if isinstance(gt_traj, torch.Tensor):
        gt_arr = gt_traj.detach().cpu().numpy()
    else:
        gt_arr = np.asarray(gt_traj)

    if isinstance(pred_traj, torch.Tensor):
        pred_arr = pred_traj.detach().cpu().numpy()
    else:
        pred_arr = np.asarray(pred_traj)

    vmin = np.min(gt_arr)
    vmax = np.max(gt_arr)

    # Animation to compare to a single trajectory
    for i in tqdm(range(n_animate)):
        if i < n_animate:
            im = axs[0].imshow(gt_arr[i, :].reshape(s, s), animated=True, cmap='RdBu', vmin=vmin, vmax=vmax)
            im2 = axs[1].imshow(pred_arr[i, :].reshape(s, s), animated=True, cmap='RdBu', vmin=vmin, vmax=vmax)
            ims.append([im, im2])

    ani = animation.ArtistAnimation(fig,ims,interval = 1e-6)
    print('saving animation')
    update_func = lambda _i, _n: progress_bar.update(1)
    with tqdm(total=n_animate, desc='Saving video') as progress_bar:
        ani.save(f"{figs_dir}/{savename}.gif",progress_callback=update_func, writer = 'ffmpeg', fps=fps)
    return None

def plot_cos_sims(
    dt,
    trajs,
    pred_trajs,
    traj_length=None,
    save_path=None,
    thresholds=(0.9, 0.8),
):
    """
    Plot average cosine similarity between ground-truth and predicted trajectories.

    Args:
        dt (float): Time step between successive states.
        trajs (Tensor or ndarray): Ground-truth states shaped (T, D) or (B, T, D).
        pred_trajs (Tensor or ndarray): Predicted states shaped like trajs.
        traj_length (int, optional): Number of time steps to include. If None use min length.
        save_path (str, optional): If provided, saves the plot and closes the figure.
        thresholds (tuple): Horizontal reference lines to draw (default 0.9 and 0.8).

    Returns:
        np.ndarray: Cosine similarities for each time step (length = traj_length).
        matplotlib.figure.Figure | None: The created figure if `save_path` is None.

    Notes:
        When multiple trajectories are provided, each time slice is mean-centered across the batch
        before cosine similarities are computed, matching the JAX implementation's batch averaging.
    """
    true = torch.as_tensor(trajs, dtype=torch.float32).detach()
    pred = torch.as_tensor(pred_trajs, dtype=torch.float32).detach()

    if true.ndim == 2:
        true = true.unsqueeze(0)
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)

    if true.shape[-1] != pred.shape[-1]:
        raise ValueError(f"State dimensions do not match: {true.shape} vs {pred.shape}")

    if traj_length is None:
        traj_length = min(true.shape[1], pred.shape[1])
    traj_length = min(traj_length, true.shape[1], pred.shape[1])
    if traj_length <= 0:
        raise ValueError("traj_length must be positive.")

    true = true[:, :traj_length, :].to(torch.float64)
    pred = pred[:, :traj_length, :].to(torch.float64)

    if true.shape[0] > 1:
        true = true - true.mean(dim=0, keepdim=True)
    if pred.shape[0] > 1:
        pred = pred - pred.mean(dim=0, keepdim=True)

    true_norm = true / true.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    pred_norm = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-9)

    cosine_sims = (true_norm * pred_norm).sum(dim=-1).mean(dim=0).cpu().numpy()
    plot_time = np.arange(traj_length) * float(dt)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), tight_layout=True)
    colors = ["black", "red"]
    labels = [f"{val:.1f} threshold" for val in thresholds]
    for thresh, color, label in zip(thresholds, colors, labels):
        ax.plot(plot_time, np.full_like(plot_time, thresh), color=color, linestyle="dashed", label=label)

    ax.plot(plot_time, cosine_sims, label="Avg. cosine sim.")
    ax.set_xlim([0, float(plot_time.max()) if plot_time.size > 0 else 1.0])
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("Avg. cosine sim.")
    ax.set_title("Cosine Similarity over time")
    ax.legend(frameon=False, bbox_to_anchor=(1, 1))

    if save_path is not None:
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return cosine_sims

    return cosine_sims, fig


def _compute_covariance(trajs: torch.Tensor) -> torch.Tensor:
    """
    Compute empirical covariance matrix for a collection of trajectories.
    Args:
        trajs: Tensor shaped (B, T, D) or (T, D).
    Returns:
        torch.Tensor: Covariance matrix of shape (D, D) in float64 on CPU.
    """
    trajs = torch.as_tensor(trajs, dtype=torch.float64)
    if trajs.ndim == 2:
        trajs = trajs.unsqueeze(0)
    if trajs.ndim != 3:
        raise ValueError(f"Expected trajs with 2 or 3 dims, got shape {trajs.shape}")

    samples = trajs.reshape(-1, trajs.shape[-1])
    mean = samples.mean(dim=0, keepdim=True)
    centered = samples - mean
    denom = samples.shape[0]
    if denom == 0:
        raise ValueError("Cannot compute covariance with zero samples.")
    cov = centered.T @ centered / denom
    return cov


def covariance_rmse(gt_trajs, pred_trajs) -> float:
    """
    Compute covariance RMSE ratio between predicted and ground truth trajectories.
    Args:
        gt_trajs: Ground truth trajectories, shape (B, T, D).
        pred_trajs: Predicted trajectories, shape (B, T, D).
    Returns:
        float: covRMSE as defined in Eq. (25) of the referenced paper.
    """
    cov_ref = _compute_covariance(gt_trajs)
    cov_pred = _compute_covariance(pred_trajs)

    diff_norm = torch.linalg.norm(cov_pred - cov_ref, ord='fro')
    ref_norm = torch.linalg.norm(cov_ref, ord='fro')
    if ref_norm.item() == 0:
        raise ValueError("Reference covariance norm is zero; covRMSE undefined.")
    return (diff_norm / ref_norm).item()


def _autocorrelation_time(trajs, dt: float, positive_only: bool = True) -> torch.Tensor:
    """
    Compute per-pixel autocorrelation time tau given trajectories.
    Args:
        trajs: Tensor or array with shape (B, T, D) or (T, D).
        dt: Physical timestep between snapshots.
        positive_only: Sum correlations until first non-positive entry.
    Returns:
        torch.Tensor: Average tau per pixel across batch (shape D,) in float64.
    """
    data = torch.as_tensor(trajs, dtype=torch.float64)
    if data.ndim == 2:
        data = data.unsqueeze(0)
    if data.ndim != 3:
        raise ValueError(f"Expected trajectories with 2 or 3 dims, got shape {data.shape}")

    B, T, D = data.shape
    if T < 2:
        raise ValueError("Need at least two time steps to compute autocorrelation.")

    series = data.permute(0, 2, 1).reshape(-1, T)
    series = series - series.mean(dim=1, keepdim=True)

    pad_len = 2 * T
    fft_vals = torch.fft.rfft(series, n=pad_len)
    acf = torch.fft.irfft(fft_vals * torch.conj(fft_vals), n=pad_len)
    acf = acf[:, :T]

    normalization = torch.arange(T, 0, -1, device=acf.device, dtype=acf.dtype)
    acf = acf / normalization

    var = acf[:, 0].clone()
    eps = torch.tensor(1e-12, dtype=acf.dtype, device=acf.device)
    var = torch.where(var.abs() < eps, eps, var)

    rho = acf[:, 1:] / var.unsqueeze(1)

    if positive_only:
        pos_mask = rho > 0
        stop_mask = (~pos_mask).cumsum(dim=1) > 0
        valid_mask = pos_mask & ~stop_mask
        rho = rho * valid_mask.to(rho.dtype)

    sum_rho = rho.sum(dim=1)
    tau = dt * (1.0 + 2.0 * sum_rho)
    tau = torch.clamp(tau, min=0.0)

    tau = tau.reshape(B, D)
    tau_mean = tau.mean(dim=0)
    return tau_mean


def time_correlation_metric(gt_trajs, pred_trajs, dt: float, positive_only: bool = True):
    """
    Compute the Time Correlation Metric (TCM) based on autocorrelation time.
    Args:
        gt_trajs: Ground truth trajectories, shape (B, T, D).
        pred_trajs: Predicted trajectories, shape (B, T, D).
        dt: Physical timestep.
        positive_only: Whether to truncate sum at first non-positive correlation.
    Returns:
        tuple: (metric_value, tau_gt, tau_pred) where tau arrays are numpy.
    """
    tau_gt = _autocorrelation_time(gt_trajs, dt, positive_only=positive_only)
    tau_pred = _autocorrelation_time(pred_trajs, dt, positive_only=positive_only)
    metric = torch.mean(torch.abs(tau_pred - tau_gt)).item()
    return metric, tau_gt.cpu().numpy(), tau_pred.cpu().numpy()

def sinkhorn_div(
    x,
    y,
    epsilon=0.1,
    n_iters=100,
    max_samples=1024,
    seed=None,
    stable=True,
    p=2,
):
    """
    Compute the Sinkhorn divergence between two empirical distributions.

    Args:
        x (Tensor or ndarray): Samples from distribution P, shape (N, D).
        y (Tensor or ndarray): Samples from distribution Q, shape (M, D).
        epsilon (float): Entropic regularization strength.
        n_iters (int): Maximum number of Sinkhorn iterations inside POT.
        max_samples (int): Optional cap on the number of samples from each set.
        seed (int, optional): Random seed for subsampling.

    Returns:
        float: Estimated Sinkhorn divergence between x and y.

    Raises:
        ImportError: If Python Optimal Transport (POT) is not installed.
    """
    if ot is None:
        raise ImportError(
            "sinkhorn_div requires the POT library. Install with `pip install POT`."
        )

    with torch.no_grad():
        x_samples = torch.as_tensor(x, dtype=torch.float64)
        y_samples = torch.as_tensor(y, dtype=torch.float64)

        if x_samples.ndim == 1:
            x_samples = x_samples.unsqueeze(0)
        if y_samples.ndim == 1:
            y_samples = y_samples.unsqueeze(0)

        if x_samples.ndim > 2:
            x_samples = x_samples.reshape(x_samples.shape[0], -1)
        if y_samples.ndim > 2:
            y_samples = y_samples.reshape(y_samples.shape[0], -1)

        # Remove any samples that contain NaNs or Infs to avoid numerical issues
        x_mask = torch.isfinite(x_samples).all(dim=1)
        y_mask = torch.isfinite(y_samples).all(dim=1)
        x_samples = x_samples[x_mask]
        y_samples = y_samples[y_mask]

        if x_samples.numel() == 0 or y_samples.numel() == 0:
            raise ValueError("Sinkhorn divergence received empty samples after filtering invalid values.")

        if x_samples.shape[-1] != y_samples.shape[-1]:
            raise ValueError(f"Sample dimensions do not match: {x_samples.shape} vs {y_samples.shape}")

        if max_samples is not None:
            max_n = min(max_samples, x_samples.shape[0])
            max_m = min(max_samples, y_samples.shape[0])
            if seed is not None:
                g = torch.Generator(device=x_samples.device)
                g.manual_seed(seed)
            else:
                g = None
            x_idx = torch.randperm(x_samples.shape[0], generator=g)[:max_n]
            y_idx = torch.randperm(y_samples.shape[0], generator=g)[:max_m]
            x_samples = x_samples[x_idx]
            y_samples = y_samples[y_idx]

        x_samples = x_samples.cpu().numpy()
        y_samples = y_samples.cpu().numpy()

        a = np.full(x_samples.shape[0], 1.0 / x_samples.shape[0], dtype=np.float64)
        b = np.full(y_samples.shape[0], 1.0 / y_samples.shape[0], dtype=np.float64)

        if p == 1:
            metric = "euclidean"
            power = 1
        elif p == 2:
            metric = "euclidean"
            power = 2
        else:
            raise ValueError("Only p=1 or p=2 supported for Sinkhorn divergence.")

        def _sinkhorn_cost_pot(x_arr, y_arr, wa, wb):
            M = ot.dist(x_arr, y_arr, metric=metric) ** power
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                warnings.simplefilter("ignore", category=UserWarning)
                return ot.sinkhorn2(
                    wa,
                    wb,
                    M,
                    reg=epsilon,
                    numItermax=n_iters,
                    method="sinkhorn_log" if stable else "sinkhorn",
                )

        ot_xy = _sinkhorn_cost_pot(x_samples, y_samples, a, b)
        ot_xx = _sinkhorn_cost_pot(x_samples, x_samples, a, a)
        ot_yy = _sinkhorn_cost_pot(y_samples, y_samples, b, b)

        divergence = float(ot_xy - 0.5 * (ot_xx + ot_yy))
        if not np.isfinite(divergence):
            raise FloatingPointError("Sinkhorn divergence computation returned a non-finite value.")
        return divergence

def pca_modes(w_data,w_model,figs_dir,s,device):
    U,S,V = torch.svd(w_data-torch.mean(w_data,0))

    fig,axs = plt.subplots(2,5) 
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axs[i][j].imshow(V[:,i*axs.shape[1]+j].reshape(s,s).cpu().numpy(), cmap=plt.colormaps['turbo'])

    fig.suptitle('first 10 PCA modes of data',y=0.8)
    fig.tight_layout()
    plt.savefig(f'{figs_dir}/PCA_data.png')

    ## should be same as ellipsoid method...
    plt.figure()
    mode1 = V[:,0].to(device)
    mode2 = V[:,1].to(device)

    x = torch.einsum('bi,i ->b',w_data-torch.mean(w_data,0),mode1).cpu()
    y = torch.einsum('bi,i ->b',w_data-torch.mean(w_data,0),mode2).cpu()

    plt.plot(x,y,'.',label='data')

    # mode1 = V_model[:,0]
    # mode2 = V_model[:,1]

    print(mode1.device)
    print(mode2.device)
    print(w_model.device)

    x_model = torch.einsum('bi,i ->b',w_model-torch.mean(w_data,0),mode1).cpu()
    y_model = torch.einsum('bi,i ->b',w_model-torch.mean(w_data,0),mode2).cpu()
    plt.plot(x_model,y_model,'r.',label ='model')
    plt.legend()
    plt.savefig(f'{figs_dir}/PCA_compare.png')


    U,S,V = torch.svd(w_model-torch.mean(w_model,0))

    fig,axs = plt.subplots(2,5) 
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axs[i][j].imshow(V[:,i*axs.shape[1]+j].reshape(s,s).cpu().numpy(), cmap=plt.colormaps['turbo'])

    fig.suptitle('first 10 PCA modes of model',y=0.8)
    fig.tight_layout()
    plt.savefig(f'{figs_dir}/PCA_model.png')


def visualize_ellipsoid(gt_traj, pred_traj, figs_dir, Q=None, c=1.0,tag=''):
    # Perform a PCA on the reshaped data, data is of size (num_traj, traj_length, traj_dim), we can lump all trajectory together into (traj_length, traj_dim)
    reshaped_data = gt_traj.reshape(-1, gt_traj.shape[-1]).detach().cpu().numpy()
    if pred_traj is not None:
        pred_traj = pred_traj.reshape(-1, pred_traj.shape[-1]).detach().cpu().numpy()

    print(reshaped_data.shape)
    pca = PCA(n_components=2)
    pca_traj_1 = pca.fit_transform(reshaped_data)
    if pred_traj is not None:
        # pca_traj_pred = pca.fit_transform(pred_traj)
        pca_traj_pred = pca.transform(pred_traj)
    U = pca.components_

    if Q is None:
        # If no Q is provided, use the covariance
        Q = np.eye(U.shape[-1])
        
    Q_inv = np.linalg.inv(Q)
    A = np.linalg.inv(U @ Q_inv @ U.T)

    # ----------------------------
    # Extract ellipse parameters from A:
    # The ellipse in PCA space is given by y^T A y = c.
    # Its semi-axis lengths are given by sqrt(c / eigenvalue).
    # The eigenvectors determine the orientation.
    # ----------------------------
    eigvals, eigvecs = np.linalg.eigh(A)
    print(eigvals)
    # Sort eigenvalues and eigenvectors in descending order.
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Compute semi-axis lengths (for y^T A y = c)
    # print(eigvals)
    axis_length1 = np.sqrt(c**2 / eigvals[0])
    axis_length2 = np.sqrt(c**2 / eigvals[1])

    # Compute the rotation angle (in degrees) of the ellipse.
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    # ----------------------------
    # Plot the projected ellipsoid in the 2D PCA space.
    # ----------------------------
    fig, ax = plt.subplots(figsize=(10, 10))

    # For a centered ellipsoid, the projected center is U*(x0 - x0)=0.
    ellipse_patch = Ellipse(xy=(0, 0),
                            width=2 * axis_length1,   # full axis length in the first direction
                            height=2 * axis_length2,  # full axis length in the second direction
                            angle=angle,
                            edgecolor='red', facecolor='none', lw=2
                            )
    ax.add_patch(ellipse_patch)

    # Optionally, plot the PCA-transformed trajectory data for context.
    ax.scatter(pca_traj_1[:, 0], pca_traj_1[:, 1], s=2, alpha=0.3, label="PCA GT")
    if pred_traj is not None:
        ax.scatter(pca_traj_pred[:, 0], pca_traj_pred[:, 1], s=2, alpha=0.3, label="PCA Pred")
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)

    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend()
    plt.axis('equal')
    plt.savefig(f'{figs_dir}/PCA_ellipsoid_{tag}.png')
    plt.close()
    return pca_traj_1[:,:2], pca_traj_pred[:,:2]

def rollout_on_test(eval_model, data_x, trunk_scale, test_traj, device, figs_dir, project,c):
    ## CURRENTLY UNUSED, NEED TO MAKE CHANGES BEFORE USING FOR KF
    eval_model.eval()
    trunk_input = torch.tensor(data_x, dtype=torch.float32).to(device) * trunk_scale
    # make trunk input 512 by 1, now it's 512
    trunk_input = trunk_input.view(512, 1)

    # print("trunk_input shape:", trunk_input.shape)
    test_traj = torch.tensor(test_traj, dtype=torch.float32).to(device)
    pred_traj = torch.zeros_like(test_traj).to(device)
    pred_traj[:, 0, :] = test_traj[:, 0, :].to(device)

    dt = 1
    # Q = eval_model.V._construct_Q().to(device)
    V_hist = torch.zeros(test_traj.shape[1]-1).to(device)
    V_hist_GT = torch.zeros(test_traj.shape[1]-1).to(device)

    if not project:
        c = c
    else:
        c = eval_model.c 

    for t in tqdm(range(test_traj.shape[1]-1)):
    # for t in tqdm(range(150)):
        with torch.no_grad():
        # Forward pass through the model
            input_t = (pred_traj[:, t, :], trunk_input)
            pred_traj[:, t+1, :] = eval_model(input_t)

            w_in = pred_traj[0, t, :]
            w_out = pred_traj[0, t+1, :]
            # w_diff = w_in - eval_model.V.x_0

            # dVdw = 2 * (w_diff @ Q)
            # # cond = (dVdw * w_out).sum(dim=1) - (dVdw * w_in).sum(dim=1) + dt * (eval_model.V(w_in) - eval_model.c ** 2)
            # A = dVdw
            # bx = eval_model.V(w_in)-(1/dt) * torch.einsum('bi,bi->b',dVdw, w_in) - eval_model.c**2
            # cond = torch.einsum('bi,bi->b',A,w_out) + bx
            # # print((dVdw ** 2).sum(dim=1))

            # if cond > 0:
            # 	print(f"Condition violated at timestep {t}: {cond.item()}")
            # # print(cond)

            if not project:
                # If no Q is provided, use the covariance
                Q = torch.eye(trunk_input.shape[0]).to(device)

                V_hist[t] = w_in @ Q @ w_in.T #torch.einsum('bi,ij,bj->b', w_in, Q, w_in)
                V_in = w_in @ Q @ w_in.T #torch.einsum('bi,ij,bj->b', w_in, Q, w_in)
                V_out = w_out @ Q @ w_out.T #torch.einsum('bi,ij,bj->b', w_out, Q, w_out)
                V_hist_GT[t] = test_traj[:,t,:] @ Q @ test_traj[:,t,:].T
            else:
                Q = eval_model.V._construct_Q()
                V_hist[t] = eval_model.V(w_in)
                V_in = eval_model.V(w_in)
                V_out = eval_model.V(w_out)
                w0 = eval_model.V.x_0
                V_hist_GT[t] = (test_traj[:,t,:]-w0) @ Q @ (test_traj[:,t,:]-w0).T

            
            if V_in > c ** 2:
                cond = V_out - V_in
                if cond > 0:
                    print(f"Condition violated at timestep {t} OUTSIDE: {cond.item()}")
            else:
                cond = V_out - c ** 2
                if cond > 0:
                    print(f"Condition violated at timestep {t} INSIDE: {cond.item()}")

    # plot the V_hist against time
    plt.plot(V_hist.cpu().numpy(),label='model')
    plt.plot(V_hist_GT.cpu().numpy(),label='GT')

    # plot c as a single line
    # plt.plot(eval_model.c.detach().cpu().numpy() ** 2, label='c')
    plt.xlabel("Time step")
    plt.ylabel("V")
    plt.yscale("log")
    plt.title("V over time")
    plt.legend()
    plt.savefig(f'{figs_dir}/V_plot.png')
    plt.close()

    # plt.figure()
    # plt.imshow(pred_traj[0,...].T.cpu().numpy(),aspect="auto")
    # plt.savefig(f'{figs_dir}/traj_forPCA.png')

    return pred_traj



def compare_distributions(gt_traj, pred_traj, bins=50, plot=True, save_name='distribution.png'):
    """
    Compute KL divergence between two histograms (discrete distributions).
    """
    # Combine to get shared bin edges
    all_data = np.concatenate([gt_traj, pred_traj])
    bin_edges = np.linspace(np.min(all_data), np.max(all_data), bins + 1)

    # Histogram counts
    p_counts, _ = np.histogram(gt_traj, bins=bin_edges)
    q_counts, _ = np.histogram(pred_traj, bins=bin_edges)

    # Convert to probabilities
    p = p_counts / np.sum(p_counts)
    q = q_counts / np.sum(q_counts)

    # Clip to avoid log(0)
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)

    # Discrete KL divergence
    kl_div = np.sum(rel_entr(p, q))

    # Plot with seaborn
    if plot:
        plt.figure(figsize=(8, 5))
        sns.kdeplot(gt_traj, label='Ground Truth', fill=False, alpha=0.7, color='blue')
        sns.kdeplot(pred_traj, label='Model Prediction', fill=False, alpha=0.7, color='red')
        plt.title(f'Comparison (KL Divergence = {kl_div:.4f})')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_name)
        plt.close()

    return kl_div


def pca_histogram_eval(gt_pca, pred_pca, bins=50, lim=[[-50.0, 50.0], [-50.0, 50.0]], save_path=None, title_gt='GT', title_pred='Projection'):
    # ----- histogram -----
    if lim is None:          # auto-range
        mx = np.max(np.abs(np.vstack((gt_pca, pred_pca))), axis=0)
        lim = [[-mx[0], mx[0]], [-mx[1], mx[1]]]

    H_gt,  xe, ye = np.histogram2d(gt_pca[:,0],  gt_pca[:,1],
                                   bins=bins, range=lim, density=True)
    
    H_pr,  _,  _  = np.histogram2d(pred_pca[:,0], pred_pca[:,1],
                                   bins=bins, range=lim, density=True)

    # ----- KL / JS -----
    KL_pred = hists_to_KL(xe, ye, bins, H_gt, H_pr, option='KL')
    JS_pred = hists_to_KL(xe, ye, bins, H_gt, H_pr, option='JS')

    # ----- figure -----
    if save_path:
        vmax = max(H_gt.max(), H_pr.max())

        fig, ax = plt.subplots(1, 2, figsize=(18,6))
        for A, H, ttl in zip(ax, (H_gt, H_pr), (title_gt, title_pred)):
            im = A.imshow(H.T, origin='lower', extent=[xe[0],xe[-1],ye[0],ye[-1]],
                          vmin=0, vmax=vmax, aspect='auto')
            A.set_title(ttl)

        plt.tight_layout()
        
        # Make room on the right
        fig.subplots_adjust(right=0.85)

        # Create a new Axes on the right side of the figure for the colorbar
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  
        # [left, bottom, width, height] in figure coordinates

        # Now draw the colorbar on that new Axes
        cbar = fig.colorbar(im, cax=cbar_ax)
        # cbar.set_label("Density")  # Optional label

        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    return KL_pred, JS_pred 

def hists_to_KL(xedges, yedges, bins, H1, H2, eps=1e-9, option='KL'):
    bin_width_x = (xedges[-1] - xedges[0]) / bins
    bin_width_y = (yedges[-1] - yedges[0]) / bins
    bin_area = bin_width_x * bin_width_y

    H1 = H1 * bin_area
    H1 = H1 / H1.sum()
    H2 = H2 * bin_area
    H2 = H2 / H2.sum()

    h1 = H1.ravel()
    h2 = H2.ravel()

    h1 = h1 + eps
    h2 = h2 + eps
    h1 /= h1.sum()
    h2 /= h2.sum()
    
    if option == 'KL':
        result = np.sum(h1 * np.log(h1 / h2))
    elif option == 'JS':
        result = jensenshannon(h1, h2)

    return result


def fourier_spectrum_2d(gt_traj,pred_traj,s,figs_dir,device):
    u_new = gt_traj.reshape(-1,s,s).to(device)
    # u_new = u_new[:60000,...]
    print(u_new.shape)
    u_fft = torch.fft.fft2(u_new)

    k_max = u_new.shape[-1]//2
    # print(k_max)

    # torch fft returns X such that X[i] = conj(X[-i]), where i*2*pi is the frequency
    # wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1), \
    #                         torch.arange(start=-k_max, end=0, step=1)), 0).repeat(s, 1)
    wavenumbers = torch.fft.fftfreq(u_new.shape[-1],1/u_new.shape[-1]).repeat(s, 1)

    k_x = wavenumbers.t().to(device)
    k_y = wavenumbers.to(device)

    # Get wavenumbers (k_x**2 + k_y**2)
    prod_k = torch.sqrt(k_x**2+k_y**2).int().detach().cpu().numpy()


    # # Remove symmetric components from wavenumbers 
    # index = -1.0 * np.ones((s, s))
    # # print(index)
    # index[0:k_max + 1, 0:k_max + 1] = sum_k[0:k_max + 1, 0:k_max + 1]
    # # print(index)
    # # index = sum_k 

    spectrum = np.zeros((u_fft.shape[0], s))
    for j in range(1, s + 1):
        # print(j)
        ind = np.where(prod_k == j) # change index to prod_k to match bottom
        # print(ind)
        # spectrum[:, j - 1] = np.sqrt( (u_fft[:, ind[0], ind[1]].sum(axis=1)).abs() ** 2) # change this to the one below to match bottom
        spectrum[:, j - 1] =  ((u_fft[:, ind[0], ind[1]]).abs()**2).sum(axis=1).detach().cpu().numpy() 
    # print(spectrum)
    spectrum = spectrum.mean(axis=0)
    # print(spectrum.shape)

    u_new = pred_traj.reshape(-1,s,s).to(device)
    print(u_new.shape)
    u_fft = torch.fft.fft2(u_new)

    spectrum_model = np.zeros((u_fft.shape[0], s))
    for j in range(1, s + 1):
        ind = np.where(prod_k == j) # change index to prod_k to match bottom
        # print(ind)
        # spectrum_model[:, j - 1] = np.sqrt( (u_fft[:, ind[0], ind[1]].sum(axis=1)).abs() ** 2) # change this to the one below to match bottom
        spectrum_model[:, j - 1] =  ((u_fft[:, ind[0], ind[1]]).abs()**2).sum(axis=1).detach().cpu().numpy()  
    spectrum_model = spectrum_model.mean(axis=0)
    # print(spectrum.shape)



    fig, ax = plt.subplots()
    ax.plot(spectrum[:s//2-1],label = 'data')
    ax.plot(spectrum_model[:s//2-1],label = 'model')
    ax.set_yscale('log')
    plt.xlabel('wave number')
    plt.ylabel('energy')
    plt.legend()
    plt.savefig(f'{figs_dir}/energyspectrum.png')




def evaluate_fourier_spectrum(gt_traj, star_traj, save_path=None):
    """
    Computes, plots, and evaluates the Fourier power spectra of trajectories.
    Handles both single (T, D) and batched (B, T, D) trajectory inputs.
    """
    # Helper function to calculate the average power spectrum from a single trajectory
    def get_power_spectrum(traj):
        fft_coeffs = np.fft.fft(traj, axis=0)
        power = np.abs(fft_coeffs)**2
        avg_power = np.mean(power, axis=1)
        return avg_power[:len(avg_power) // 2]

    # Helper function to calculate Mean Squared Error on a log scale
    def calculate_spectrum_error(spec_true, spec_pred, eps=1e-12):
        log_spec_true = np.log10(spec_true + eps)
        log_spec_pred = np.log10(spec_pred + eps)
        return np.mean((log_spec_true - log_spec_pred)**2)

    # Check if the input is batched by checking the number of dimensions
    if gt_traj.ndim == 3:
        batch_size = gt_traj.shape[0]
        print(f"\n--- Performing Batched Fourier Evaluation ({batch_size} trajectories) ---")
        
        batch_errors_star = []
        
        # Store spectra from each batch item to average them later for plotting
        all_spec_gt = []
        all_spec_star = []

        for i in range(batch_size):
            # Get single trajectories from the batch
            gt_single = gt_traj[i]
            star_single = star_traj[i]

            # Calculate spectrum for this item
            spec_gt = get_power_spectrum(gt_single)
            spec_star = get_power_spectrum(star_single)

            # Store spectra for averaging
            all_spec_gt.append(spec_gt)
            all_spec_star.append(spec_star)
            
            # Calculate and store errors for this item
            batch_errors_star.append(calculate_spectrum_error(spec_gt, spec_star))

        # Average the errors and spectra across the entire batch
        final_error_star = np.mean(batch_errors_star)
        
        plot_spec_gt = np.mean(all_spec_gt, axis=0)
        plot_spec_star = np.mean(all_spec_star, axis=0)

    elif gt_traj.ndim == 2:
        print("\n--- Performing Single Trajectory Fourier Evaluation ---")
        # For a single trajectory, the final error is just the calculated error
        plot_spec_gt = get_power_spectrum(gt_traj)
        plot_spec_star = get_power_spectrum(star_traj)
        
        final_error_star = calculate_spectrum_error(plot_spec_gt, plot_spec_star)
    else:
        raise ValueError(f"Input trajectory has an unsupported shape: {gt_traj.shape}. Expected 2 or 3 dimensions.")

    # --- Common code for printing and plotting ---
    
    print(f"Spectrum Difference (GT vs. f* Projected): {final_error_star:.6f}")
    print("---------------------------------")
    
    if save_path:
        plt.figure(figsize=(12, 7))
        modes = np.arange(len(plot_spec_gt))

        plt.plot(modes, plot_spec_gt, label='Ground Truth', color='black', linewidth=2.5, alpha=0.8)
        plt.plot(modes, plot_spec_star, label=r'$f^*$ Projected', color='#1f77b4', linestyle='--')

        plt.yscale('log')
        plt.xlabel('Frequency Mode Index')
        plt.ylabel('Power (Log Scale)')
        plt.title('Fourier Power Spectrum Comparison')
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved Fourier spectrum plot to {save_path}")

    return final_error_star

def spatial_corr(u_data,u_model,figs_dir,s):
    print(u_data.shape)

    u_corr = u_data.reshape(-1,s,s)

    fig,axs = plt.subplots(1,2) 
    corr2 = np.zeros(u_corr.shape)
    for i in range(u_corr.shape[-1]):
        corr2[i,...] = scipy.signal.correlate2d(u_corr[i,...],u_corr[i,...],mode = 'same',boundary = 'wrap')
    # print(corr2.shape)
    corr2 = np.mean(corr2,axis=0)
    # print(corr2.shape)
    im = axs[0].imshow(corr2)
    fig.colorbar(im,shrink=0.6)
    axs[0].set_title('data')
    # plt.show()

    u_corr = u_model.reshape(-1,s,s)
    corr2 = np.zeros(u_corr.shape)
    for i in range(u_corr.shape[-1]):
        corr2[i,...] = scipy.signal.correlate2d(u_corr[i,...],u_corr[i,...],mode = 'same',boundary = 'wrap')
    # print(corr2.shape)
    corr2 = np.mean(corr2,axis=0)
    # print(corr2.shape)
    im = axs[1].imshow(corr2)
    fig.colorbar(im,shrink=0.6)
    axs[1].set_title('model')
    plt.suptitle('spatial correlation',y=0.8)
    plt.tight_layout()
    plt.savefig(f'{figs_dir}/spatialcorr.png')


# Training KF normalizing and de-normalizing util functions
class Normalizer:
    """Scalar (global) z-score normalizer for fields."""
    def __init__(self, mu=None, sigma=None, eps=1e-6):
        self.mu = None if mu is None else float(mu)
        self.sigma = None if sigma is None else float(sigma)
        self.eps = eps

    def fit(self, t: torch.Tensor):
        mu = t.mean()
        sigma = t.std().clamp_min(self.eps)
        self.mu = float(mu)
        self.sigma = float(sigma)
        return self

    def _like(self, t: torch.Tensor):
        return (torch.tensor(self.mu, dtype=t.dtype, device=t.device),
                torch.tensor(self.sigma, dtype=t.dtype, device=t.device))

    def norm(self, t: torch.Tensor):
        if self.mu is None:  # identity if not fitted / disabled
            return t
        mu_t, sigma_t = self._like(t)
        return (t - mu_t) / sigma_t

    def denorm(self, t: torch.Tensor):
        if self.mu is None:
            return t
        mu_t, sigma_t = self._like(t)
        return t * sigma_t + mu_t


class InferenceWrapper(torch.nn.Module):
    """
    Wraps the trained model so you can call it with PHYSICAL inputs and
    get PHYSICAL outputs, even if the model was trained in normalized space.
    Handles residual mode too.
    """
    def __init__(self, base_model: torch.nn.Module, normalizer: Normalizer, residual: bool):
        super().__init__()
        self.base = base_model
        self.norm = normalizer
        self.residual = residual

    @torch.no_grad()
    def forward(self, x):
        # x = (u_t_phys, trunk)
        u_t_phys, trunk = x
        u_t_n = self.norm.norm(u_t_phys)          # normalize input
        pred_n = self.base((u_t_n, trunk))        # model in normalized space
        if self.residual:
            next_n = u_t_n + pred_n               # delta -> next (still normalized)
        else:
            next_n = pred_n
        next_phys = self.norm.denorm(next_n)      # back to physical
        return next_phys
def energy_time(gt_traj, pred_traj, c=100.0, model=None, figs_dir=""):
    if torch.is_tensor(gt_traj):
        gt = gt_traj.detach().cpu()
    else:
        gt = torch.as_tensor(gt_traj)

    if torch.is_tensor(pred_traj):
        pred = pred_traj.detach().cpu()
    else:
        pred = torch.as_tensor(pred_traj)

    model_device = None
    if model is not None:
        model_device = next(model.parameters()).device

    V_hist = torch.zeros(pred.shape[0], dtype=torch.float64)
    V_hist_GT = torch.zeros(pred.shape[0], dtype=torch.float64)

    if model is None or not getattr(model, "project", False):
        Q_diag = torch.ones(gt.shape[-1], dtype=torch.float64)
        w0 = torch.zeros_like(Q_diag)
    else:
        Q_diag = torch.diag(model.V._construct_Q()).detach().cpu().to(torch.float64)
        w0 = model.V.x_0.detach().cpu().to(torch.float64)

    with torch.no_grad():
        for t in range(pred.shape[0]):
            w_in = pred[t].to(torch.float64)
            if model is not None and getattr(model, "project", False):
                w_in_dev = w_in.to(model_device)
                V_hist[t] = model.V(w_in_dev).detach().cpu().to(torch.float64)
            else:
                V_hist[t] = torch.dot(w_in, w_in)

            if model is not None and getattr(model, "project", False):
                diff = gt[t].to(torch.float64) - w0
                V_hist_GT[t] = torch.sum(diff * diff * Q_diag)
            else:
                gt_vec = gt[t].to(torch.float64)
                V_hist_GT[t] = torch.dot(gt_vec, gt_vec)

    plt.figure()
    plt.plot(V_hist.numpy(), label="model")
    plt.plot(V_hist_GT.numpy(), label="GT")

    if model is not None and getattr(model, "project", False):
        c_val = float(model.c.detach().cpu().item())
    else:
        c_val = c
    plt.plot(np.array([0, V_hist.shape[-1]]), np.ones(2) * (c_val ** 2), label="c")

    plt.xlabel("Sample")
    plt.ylabel("V")
    plt.yscale("log")
    # plt.title("V over time")
    plt.legend()
    plt.savefig(f'{figs_dir}/V_plot.png')
    plt.close()

    # plt.figure()
    # plt.imshow(pred_traj[0,...].T.cpu().numpy(),aspect="auto")
    # plt.savefig(f'{figs_dir}/traj_forPCA.png')

    return pred_traj


def animate_saved_rollout(
    pred_traj_path: str,
    gt_seq: torch.Tensor,
    figs_dir: str,
    s: int,
    out_name: str = "rollout.gif",
    max_frames: int = None,
    stride: int = 1,
    fps: int = 20,
):
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    device = torch.device('cpu')  # render on CPU
    pred_all = torch.load(pred_traj_path, map_location=device)
    if isinstance(pred_all, np.ndarray):
        pred_all = torch.tensor(pred_all)

    # Align lengths: prediction index 1.. matches gt 0..
    T = min(gt_seq.shape[0], pred_all.shape[0] - 1)
    gt = gt_seq[:T].detach().to(device)
    pred = pred_all[1:1+T].detach().to(device)

    # Frame thinning
    frame_idx = np.arange(0, T, stride)
    if max_frames is not None:
        frame_idx = frame_idx[:max_frames]
    if len(frame_idx) == 0:
        raise ValueError("No frames selected; adjust stride/max_frames.")
    T_eff = len(frame_idx)

    # ---- FIX: robust vmin/vmax via NumPy (nan-safe) ----
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    all_np = np.concatenate([gt_np, pred_np], axis=0)  # shape (2T, s*s)
    vmin = float(np.nanmin(all_np))
    vmax = float(np.nanmax(all_np))
    # ----------------------------------------------------

    fig, axs = plt.subplots(2, 1, figsize=(5, 6))
    axs[0].set_title('Data')
    axs[1].set_title('Model')
    fig.tight_layout()

    im_data = axs[0].imshow(gt[frame_idx[0]].reshape(s, s).cpu().numpy(),
                            animated=True, cmap='RdBu', vmin=vmin, vmax=vmax)
    im_pred = axs[1].imshow(pred[frame_idx[0]].reshape(s, s).cpu().numpy(),
                            animated=True, cmap='RdBu', vmin=vmin, vmax=vmax)
    txt = fig.text(0.02, 0.02, f"t={frame_idx[0]}", fontsize=10)

    def _update(k):
        i = frame_idx[k]
        im_data.set_array(gt[i].reshape(s, s).cpu().numpy())
        im_pred.set_array(pred[i].reshape(s, s).cpu().numpy())
        txt.set_text(f"t={i}")
        return im_data, im_pred, txt

    ani = animation.FuncAnimation(fig, _update, frames=T_eff, blit=False, interval=50)

    save_path = f"{figs_dir}/{out_name}"
    ani.save(save_path, writer='ffmpeg', fps=fps)
    plt.close(fig)
    print(f"Saved animation to: {save_path}")
