import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm
from scipy.integrate import solve_ivp

def tensor_lorenz63(X, sigma=10, beta=8/3, rho=28):
    """
    X: pytorch tensor of shape (M, 3) is the batch of states in M different trajectories
    """
    assert(X.shape[1] == 3)
    dX = torch.zeros_like(X)
    dX[:, 0] = sigma * (X[:, 1] - X[:, 0])
    dX[:, 1] = X[:, 0] * (rho - X[:, 2]) - X[:, 1]
    dX[:, 2] = X[:, 0] * X[:, 1] - beta * X[:, 2]
    return dX

def rk4_model(func, dt, y, **kwargs):
    f1 = func(y, **kwargs)
    f2 = func(y + dt / 2 * f1, **kwargs)
    f3 = func(y + dt / 2 * f2, **kwargs)
    f4 = func(y + dt * f3, **kwargs)
    y1 = y + dt / 6 * (f1 + 2 * f2 + 2 * f3 + f4)
    return y1

def gen_real_multi_traj(M, N, dt, dt_target=1.0, seed=0, odefun=tensor_lorenz63, x_lim=50.0, x_dim=3, **kwargs):
    # Check if GPU is available and set the appropriate device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(seed)  # Set random seed for reproducibility
    
    # Create initial tensor and move it to the device
    X0 = (torch.rand(M, x_dim, device=device) * 2 * x_lim) - x_lim  # X0 initialized on GPU/CPU

    N_data = int(N * dt_target / dt)
    X = torch.zeros(M, N_data+1, x_dim, device=device)  # Initialize X on GPU/CPU

    # Set initial condition
    X[:, 0, :] = X0
    
    # Iterate and compute trajectory
    for i in tqdm(range(N_data)):
        X[:, i + 1, :] = rk4_model(odefun, dt, y=X[:, i, :], **kwargs)
        if torch.isnan(X[:, i + 1, :]).any():
            raise ValueError(f"NaN encountered at time step {i+1}")

    sample_rate = int(dt_target / dt)
    X_ds = X[:, ::sample_rate, :].detach().cpu().numpy()  # Downsample and move to CPU

    return X_ds

def lorenz63_np(t, u, sigma=10, rho=28, beta=8/3):
    """Lorenz-63 ODEs for use with SciPy's solve_ivp."""
    dudt = np.zeros_like(u)
    dudt[0] = sigma * (u[1] - u[0])
    dudt[1] = u[0] * (rho - u[2]) - u[1]
    dudt[2] = u[0] * u[1] - beta * u[2]
    return dudt

def gen_multi_traj_scipy(M, N, dt_target, seed=0, x_lim=50.0, x_dim=3, **kwargs):
    """Generates multiple trajectories using scipy's RK45 solver."""
    np.random.seed(seed)
    X0 = (np.random.rand(M, x_dim) * 2 * x_lim) - x_lim
    
    t_span = [0, N * dt_target]
    t_eval = np.arange(0, t_span[1], dt_target)
    
    trajectories = []
    for i in tqdm(range(M)):
        sol = solve_ivp(lorenz63_np, t_span, X0[i], t_eval=t_eval, **kwargs)
        trajectories.append(sol.y.T)
        
    return np.stack(trajectories, axis=0)

class TrajectoryTensorDataset(Dataset):
    # Currently, this class is only meant for real-valued trajectories. For complex trajectories such as the order-4 truncated Kuramoto-Sivashinsky equation, we need to preprocess the complex-valued trajectories into its real and imaginary tensor parts.
    def __init__(self, trajectories, subtraj_length, stride):
        self.trajectories = trajectories
        self.subtraj_length = subtraj_length
        self.stride = stride
        self.indices = []
        
        num_traj, traj_length, _ = trajectories.shape
        for traj_index in range(num_traj):
            for start in range(0, traj_length - subtraj_length + 1, stride):
                self.indices.append((traj_index, start))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        traj_index, start = self.indices[idx]
        return self.trajectories[traj_index, start:start + self.subtraj_length, :]