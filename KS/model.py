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
class V_elliptical(nn.Module):
    def __init__(self, m):
        super(V_elliptical, self).__init__()

        self.latent_dim = m
        # diagonal elements of the lower triangular matrix L
        self.log_diag_L = nn.Parameter(torch.zeros(self.latent_dim))
        self.log_diag_L = nn.Parameter(torch.ones(self.latent_dim)*0.1)

        # 2. Learnable parameters for the strictly lower triangular (off-diagonal) elements of L.
        # Get the indices for the lower triangular part of an n x n matrix (excluding the diagonal).
        tril_indices = torch.tril_indices(row=self.latent_dim, col=self.latent_dim, offset=-1)
        self.off_diag_L = nn.Parameter(torch.randn(len(tril_indices[0])) * 0.1) # Initialize with small random values

        # We store the indices as a buffer, so they are part of the model's state but not its parameters.
        self.register_buffer('tril_indices', tril_indices)
    
        # Trainable vector x_0
        self.x_0 = nn.Parameter(torch.randn(1, m))

        self.Q = None  # Placeholder for the symmetric positive-definite matrix Q``

    def _construct_Q(self):
        """
        Constructs the symmetric positive-definite matrix V_elliptical (Q) from L.
        """
        # Create an empty n x n matrix for L
        L = torch.zeros(self.latent_dim, self.latent_dim, device=self.log_diag_L.device)

        # Set the diagonal elements using the log_diag_L parameters.
        # The exp() ensures the diagonal is always positive. **** positive diagonal means L is a unique solution to A = LLT. that way we aren't getting the same Q with different L's (redundant, probably confusing during training)
        L.diagonal().copy_(torch.exp(self.log_diag_L))

        # Set the off-diagonal elements from the learned parameters.
        L[self.tril_indices[0], self.tril_indices[1]] = self.off_diag_L

        # Compute Q = LLᵀ
        Q = torch.matmul(L, L.T)
        return Q

        
    def forward(self, x):
        Q = self._construct_Q()

        self.Q = Q
        
        # # Reshape x_0 to broadcast correctly
        # x_0 = self.x_0.squeeze(-1)
        # Calculate (x - x_0)
        diff = x - self.x_0
        
        # Calculate V for each input in the batch
        V = torch.einsum('bi,ij,bj->b', diff, Q, diff)
        # V = V.unsqueeze(1)
        return V



class DeepONet(nn.Module):
    def __init__(self,m,n,project=False):
        super(DeepONet,self).__init__()

        self.Branch = Branch(m)
        self.Trunk = Trunk(n)
        self.project = project

        self.b = torch.nn.Parameter(torch.tensor(0.0))
        if self.project:
            print('Projection layer included')
            self.c = torch.nn.Parameter(torch.tensor(1.0))
            self.eps_proj = 1e-3
            self.V = V_elliptical(m)
    

    def f_project(self,w_in,w_out,dt=1):
        w0 = self.V.x_0
        V = self.V(w_in)
        Q = self.V.Q
        diff = w_in-w0
        dVdw = torch.einsum('ij,bj->bi',2*Q,diff)

        # constraint has the form Ay + b(x) <= 0
        A = dVdw*(1/dt)
        bx = V-(1/dt) * torch.einsum('bi,bi->b',dVdw, w_in) - self.c**2
        w_star = w_out - dVdw * (F.relu( torch.einsum('bi,bi->b',A,w_out) + bx)/torch.clamp((dVdw**2).sum(dim=1), min=self.eps_proj)).unsqueeze(1)

        return w_star


    def forward(self,x):
        x1 = self.Branch(x[0])
        x2 = self.Trunk(x[1])
        x_out = torch.einsum("bi,ai->ba",x1,x2)
        x_out += self.b
        if self.project:
            x_out = self.f_project(x[0],x_out)
        return x_out



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
    plt.savefig(f'figs_{figs_dir}/1step.png')
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
    plt.savefig(f'figs_{figs_dir}/rollout_test.png')
    plt.close()

    # --- 3. Rollout from random initial condition ---
    key = jax.random.PRNGKey(random_seed) * 5
    u0 = jax.random.normal(key, (1, s))
    print("Random IC shape:", u0.shape)

    rollout_traj = torch.zeros(rollout_steps_random, s)
    u_out = torch.tensor(np.array(u0)).to(device)

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
    plt.savefig(f'figs_{figs_dir}/rollout_randomIC.png')
    plt.close()
