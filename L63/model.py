import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class V_elliptical(nn.Module):
    def __init__(self, m, diag_flag, x_0):
        super(V_elliptical, self).__init__()
        self.latent_dim = m
        self.diag_Q = diag_flag
        if self.diag_Q:
            print("V_elliptical initialized with a DIAGONAL Q.")
        else:
            print("V_elliptical initialized with a FULL Q.")
        
        # Diagonal elements of lower-triangular L (log-parametrized for positivity)
        self.log_diag_L = nn.Parameter(torch.zeros(self.latent_dim))

        # Off-diagonal (strictly lower) elements of L (used only if diag_Q == False)
        tril_indices = torch.tril_indices(row=self.latent_dim, col=self.latent_dim, offset=-1)
        self.off_diag_L = nn.Parameter(torch.randn(len(tril_indices[0])) * 0.1)
        self.register_buffer('tril_indices', tril_indices)
    
        # Trainable center x_0
        self.x_0 = nn.Parameter(torch.tensor(x_0, dtype=torch.float32))

        self.Q = None  # Cached SPD matrix

    def _build_L(self):
        L = torch.zeros(self.latent_dim, self.latent_dim, device=self.log_diag_L.device)
        L.diagonal().copy_(torch.exp(self.log_diag_L))
        if not self.diag_Q:
          L[self.tril_indices[0], self.tril_indices[1]] = self.off_diag_L
        return L
    def _construct_Q(self):
        # L = torch.zeros(self.latent_dim, self.latent_dim, device=self.log_diag_L.device)
        # L.diagonal().copy_(torch.exp(self.log_diag_L))
        # if not self.diag_Q:
        #     L[self.tril_indices[0], self.tril_indices[1]] = self.off_diag_L
        self.L = self._build_L()
        Q = torch.matmul(self.L, self.L.T)  # SPD
        return Q

    def forward(self, x):
        Q = self._construct_Q()
        self.Q = Q
        diff = x - self.x_0
        V = torch.einsum('bi,ij,bj->b', diff, Q, diff)
        return V


class MLP(nn.Module):
    """Standard MLP that maps (b, d) -> (b, d)."""
    def __init__(self, input_dim, hidden_dims=(128, 128), activation=nn.ReLU()):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [input_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ProjectedMLP(nn.Module):
    """
    Replacement for DeepONet:
      - Backbone: standard MLP (b, d) -> (b, d)
      - Keeps: elliptical V function and discrete/continuous projection
    """
    def __init__(self, model_params):
        super().__init__()
        d = model_params['d']               # input/output dimensionality
        # Set default values
        hidden_dims = model_params.get('hidden_dims', [128, 128])
        activation = model_params.get('activation', nn.GELU())
        self.discrete_proj = model_params.get('discrete_proj', True)
        x0_init = model_params.get('x_0', None)

        self.c0 = model_params.get('c0', 1.0)
        self.trainable_c = model_params.get('trainable_c', False)
        self.diag_Q = model_params.get('diag_Q', False)
        print(f'DIAG_Q: {self.diag_Q}')

        # Backbone
        self.mlp = MLP(input_dim=d, hidden_dims=hidden_dims, activation=activation)

        # Projection params / energy function
        self.active_projection_percentage = 0.0
        if self.discrete_proj:
            print('Projection layer included')
            self.c = nn.Parameter(torch.tensor(self.c0, dtype=torch.float32))
            self.c.requires_grad = bool(self.trainable_c)
            self.eps_proj = 1e-3
            self.V = V_elliptical(m=d, diag_flag=self.diag_Q, x_0=x0_init if x0_init is not None else np.zeros(d))

    @torch.no_grad()
    def _q_inv_sqrt_for_diag(self):
        """
        Helper for discrete projection assuming diagonal Q.
        Uses L from Q = L L^T; for diagonal Q, L is diagonal with exp(log_diag_L).
        """
        L_diag = torch.exp(self.V.log_diag_L)  # (d,)
        return torch.diag(1.0 / L_diag)        # (d, d)

    def discrete_project(self, w_in, w_out, smooth_choice=True, scale_level_set=0.999):
        """
        Discrete projection onto (w - x0)^T Q (w - x0) <= b,
        with b = V(w_in) + ReLU(-V(w_in) + c^2), then scaled.
        Assumes diagonal Q (same as your original note).
        """
        w_0 = self.V.x_0
        V_in = self.V(w_in)

        # b = (1 - gamma) * V + gamma * c^2, with your smooth max: V + ReLU(-V + c^2)
        b = V_in + F.relu(-V_in + self.c ** 2)
        b = scale_level_set * b  # tighten a bit
        w = w_out - w_0

        # Normalize direction and scale to the boundary defined by Q

        w_norms = torch.linalg.norm(w, dim=1, keepdim=True).clamp_min(1e-8)
        z = w / w_norms
        sqrt_b = torch.sqrt(b).unsqueeze(1)       # (b, 1)


        # Using Q^{-1/2} from diagonal L
        if self.V.diag_Q:
            Q_inv_sqrt = self._q_inv_sqrt_for_diag()  # (d, d)
            w_proj = w_0 + sqrt_b * (z @ Q_inv_sqrt)  # (b, d)
        else:
            L = self.V.L
            # use solve rather than directly computing inverse for speedup
            w_proj = w_0 + sqrt_b * (torch.linalg.solve(L.T,z.T)).T 
            
        # w_proj = w_0 + sqrt_b * (z @ Q_inv_sqrt)  # (b, d)

        # Soft/hard choice to keep or project
        V_out = self.V(w_out)
        if smooth_choice:
            k_choice = 100.0
            choice = 1 - torch.sigmoid(k_choice * (V_out - b))
        else:
            choice = (V_out <= b).float()
        choice = choice.reshape(-1, 1)

        # Track percentage projected (near-zero choice => projected)
        active_threshold = 1e-5
        active_proj_count = torch.sum(choice < active_threshold)
        batch_size = w_in.shape[0]
        self.active_projection_percentage = active_proj_count.item() / batch_size * 100

        w_star = choice * w_out + (1 - choice) * w_proj
        return w_star

    def forward(self, x):
        """
        x: (b, d)
        returns: (b, d)
        """
        x_out = self.mlp(x)
        if self.discrete_proj:
            x_out = self.discrete_project(x, x_out)
        return x_out
