import numpy as np

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
        if x_data.ndim == 1:
            x_data = x_data.reshape(-1, 1)

        self.trunk_input = torch.tensor(x_data, dtype=torch.float32)
        
        branch_inputs = []
        targets = []
        
        # Iterate over each trajectory to create (u_t, u_{t+1}) pairs
        for traj in u_data:
            # A trajectory of length L has L-1 possible pairs
            for t in range(traj.shape[0] - 1):
                branch_inputs.append(traj[t])
                targets.append(traj[t+1])
                
        # Convert the lists of individual steps into single large tensors for efficiency.
        self.branch_inputs = torch.tensor(np.array(branch_inputs), dtype=torch.float32)
        self.targets = torch.tensor(np.array(targets), dtype=torch.float32)
        
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
    
def load_multi_traj_data(data,trunk_scale=1.0):
    u_all_traj = data['u_batch']
    x_trunk_input = data['x']*trunk_scale
    num_traj = u_all_traj.shape[0]

    # Split trajectories into training and validation sets (80/20)
    # It's important to split trajectories, not individual time-steps,
    # to prevent data leakage between train and validation.
    num_train_traj = int(0.8 * num_traj)

    u_train_traj = u_all_traj[:num_train_traj]
    u_val_traj = u_all_traj[num_train_traj:]

    train_dataset = TrajectoryDataset(u_data=u_train_traj, x_data=x_trunk_input)
    val_dataset = TrajectoryDataset(u_data=u_val_traj, x_data=x_trunk_input)

    return train_dataset, val_dataset


def run_model_visualization(
    model,
    x_test,
    y_test,
    s,
    device,
    figs_dir='figs2',
    figs_tag = '',
    rollout_steps_test=1000,
    rollout_steps_random=10000,
    random_seed=10,
):

    # --- 1. One-step prediction visualization ---
    u_test = model(x_test)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    # print("y_test shape:", y_test.shape)
    # print("u_test shape:", u_test.shape)

    # extent = [physical_t[90000] * dt, physical_t[-1] * dt, physical_x[0], physical_x[-1]]

    axs[0].imshow(
        y_test.T.detach().cpu().numpy().astype(np.float32),
        #extent=extent,
        aspect='auto',
        vmin=-2.5,
        vmax=2.5,
    )
    axs[0].set_title('Ground Truth')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Position')

    im = axs[1].imshow(
        u_test.T.detach().cpu().numpy().astype(np.float32),
        #extent=extent,
        aspect='auto',
        vmin=-2.5,
        vmax=2.5,
    )
    axs[1].set_title('Model Prediction')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Position')

    fig.tight_layout()
    fig.colorbar(im, ax=axs, location='right')
    plt.savefig(f'{figs_dir}/1step.png')
    plt.close(fig)

    # --- 2. Long trajectory rollout (on test data) ---
    rollout_traj = torch.zeros(rollout_steps_test, s)
    u_out = x_test[0][0, ...]  # initial condition
    u_out = u_out.unsqueeze(0)

    for i in range(rollout_steps_test):
        u_out = model((u_out, x_test[1]))
        rollout_traj[i, :] = u_out

    plt.figure()
    plt.imshow(
        rollout_traj.T.detach().numpy().astype(np.float32),
        extent=[0, rollout_steps_test, 0, s],
        aspect='auto'
    )
    plt.title('Rollout on Test Data')
    plt.colorbar()
    plt.savefig(f'{figs_dir}/rollout_test.png')
    plt.close()

    # --- 3. Rollout from random initial condition ---

    # use a pytorch seed
    torch.manual_seed(random_seed)
    u0 = torch.randn(1, s).to(device)

    print("Random IC shape:", u0.shape)

    rollout_traj = torch.zeros(rollout_steps_random, s)
    u_out = u0

    for i in range(rollout_steps_random):
        u_out = model((u_out, x_test[1]))
        rollout_traj[i, :] = u_out

    plt.figure()
    plt.imshow(
        rollout_traj.T.detach().numpy().astype(np.float32),
        extent=[0, rollout_steps_random, 0, s],
        aspect='auto'
    )
    plt.title('Rollout from Random Initial Condition')
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.savefig(f'{figs_dir}/rollout_randomIC.png')
    plt.close()
