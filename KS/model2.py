import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import jax

class Branch(nn.Module):
    def __init__(self, m, activation=F.relu):
        super(Branch, self).__init__()
        self.m = m
        self.activation = activation

        # self.reshape = lambda x: x.view(-1, 1, 28, 28)
        self.reshape = lambda x: x.view(-1, 1, m)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=2)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=2)
        self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(128 * 4 * 4, 128) 
        # self.fc1 = nn.Linear(1280, 256) 
        # self.fc1 = nn.Linear(1664, 256) 
        # self.fc1 = nn.Linear(1856, 256) 
        # self.fc1 = nn.Linear(1984, 256) 
        self.fc1 = nn.Linear(128,128)
        self.fc2 = nn.Linear(128, 128)

    def forward(self, x):
        x = self.reshape(x)
        # x = self.activation(self.conv1(x))
        # x = self.activation(self.conv2(x))
        # x = self.activation(self.conv3(x))
        # x = self.activation(self.conv4(x))
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

class Trunk(nn.Module):
    def __init__(self, n, activation=F.relu):
        super(Trunk, self).__init__()
        self.n = n
        self.activation = activation

        self.fc1 = nn.Linear(n, 128) 
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        # x = self.fc2(x)
        # x = self.activation(x)
        # x = self.fc3(x)
        # x = self.activation(x)
        x = self.fc4(x)
        x = self.activation(x)
        return x

class DeepONet(nn.Module):
    def __init__(self,m,n):
        super(DeepONet,self).__init__()

        self.Branch = Branch(m)
        self.Trunk = Trunk(n)

        self.b = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self,x):
        x1 = self.Branch(x[0])
        x2 = self.Trunk(x[1])
        x_out = torch.einsum("ai,bi->ab",x1,x2)
        x_out += self.b
        return x_out



def run_model_visualization(
    model,
    x_test,
    y_test,
    physical_t,
    physical_x,
    dt,
    s,
    figs_dir='figs2',
    rollout_steps_test=1000,
    rollout_steps_random=10000,
    random_seed=10,
):

    # --- 1. One-step prediction visualization ---
    u_test = model(x_test)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    # print("y_test shape:", y_test.shape)
    # print("u_test shape:", u_test.shape)

    extent = [physical_t[90000] * dt, physical_t[-1] * dt, physical_x[0], physical_x[-1]]

    axs[0].imshow(
        y_test.T.detach().cpu().numpy().astype(np.float32),
        extent=extent,
        aspect='auto',
        vmin=-2.5,
        vmax=2.5,
    )
    axs[0].set_title('Ground Truth')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Position')

    im = axs[1].imshow(
        u_test.T.detach().cpu().numpy().astype(np.float32),
        extent=extent,
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
    key = jax.random.PRNGKey(random_seed) * 5
    u0 = jax.random.normal(key, (1, s))
    print("Random IC shape:", u0.shape)

    rollout_traj = torch.zeros(rollout_steps_random, s)
    u_out = torch.tensor(np.array(u0)).to(torch.device('cuda'))

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
