import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fno_2d import *

class Branch(nn.Module):
    def __init__(self, m, conv_config, fc_dims, output_dim=128, activation=nn.ReLU()):#, circ_padding=False):
        super(Branch, self).__init__()
        self.activation = activation
        self.reshape = lambda x: x.view(-1, 1, m, m)
        # self.circ_padding = circ_padding

        # Build the Convolutional Part  
        conv_layers = []
        in_channels = 1
        for cfg in conv_config:
            # if self.circ_padding:
            #     conv_layers.append(
            #         nn.Conv2d(
            #             in_channels=in_channels,
            #             out_channels=cfg['out_channels'],
            #             kernel_size=cfg['kernel_size'],
            #             stride=cfg['stride'],
            #             padding=cfg['kernel_size'] // 2,   # "same" spatial size
            #             padding_mode='circular',           # periodic BCs (Kolmogorov flow)
            #             bias=False                         # BN handles bias
            #         )
            #     )
            #     conv_layers.append(nn.BatchNorm2d(cfg['out_channels']))
            # else:
            conv_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=cfg['out_channels'],
                    kernel_size=cfg['kernel_size'],
                    stride=cfg['stride']
                )
            )
                
            conv_layers.append(self.activation)
            in_channels = cfg['out_channels'] # Update for the next layer
        
        self.conv_net = nn.Sequential(*conv_layers)

        # Use a Dummy Forward Pass to Find the Flattened Size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, m, m) # Batch size of 1, 1 channel
            dummy_output = self.conv_net(dummy_input)
            flattened_size = dummy_output.flatten(1).shape[1]
            print(f"Auto-detected flattened size for FC layer: {flattened_size}")

        # Build the Fully-Connected Part 
        all_fc_dims = [flattened_size] + fc_dims + [output_dim]
        fc_layers = []
        for i in range(len(all_fc_dims) - 1):
            fc_layers.append(nn.Linear(all_fc_dims[i], all_fc_dims[i+1]))
            if i < len(all_fc_dims) - 2: # No activation on the final output
                fc_layers.append(self.activation)
        
        self.fc_net = nn.Sequential(*fc_layers)


    def forward(self, x):
        x = self.reshape(x)
        x = self.conv_net(x)
        x = x.flatten(1) # Flatten all dimensions except batch
        x = self.fc_net(x)
        return x

class Trunk(nn.Module):
    def __init__(self, n, hidden_dims, output_dim=128, activation=nn.ReLU(), last_act=False):
        super(Trunk, self).__init__()
        self.activation = activation
        self.last_act = last_act

        # Create a list of all layer dimensions
        all_dims = [n] + hidden_dims + [output_dim]
        
        layers = []
        for i in range(len(all_dims) - 1):
            layers.append(nn.Linear(all_dims[i], all_dims[i+1]))
            # Add activation to all layers except the last one
            if i < len(all_dims) - 2:
                layers.append(self.activation)
            else:
                if self.last_act:
                    layers.append(self.activation)
        
        # Create the sequential model
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Q_func(nn.Module):
    def __init__(self): 
        super(Q_func, self).__init__()
        print('INITIALIZING Q BASED ON A FUNCTION')
        self.activation = nn.ReLU()
        layers = []
        layers.append(nn.Linear(2,128))
        layers.append(self.activation)
        layers.append(nn.Linear(128,1))
        self.net = nn.Sequential(*layers)
        self.initialized_output = False

    def forward(self,x):
        dx = x[1,0]-x[0,0] # assumes x and y are evenly spaced and the same
        if not self.initialized_output:
            with torch.no_grad():
                nn.init.zeros_(self.net[-1].weight)
                self.net[-1].bias.fill_(-torch.log(dx**2).item()) # initialize to 1s on the diagonal
            self.initialized_output = True
        logQ = self.net(x)
        logQ = torch.clip(logQ,min=-10) # clip so entries of Q must be nonzero
        return torch.exp(logQ)*dx*dx 

class x0_func(nn.Module):
    def __init__(self): 
        super(x0_func, self).__init__()
        print('INITALIZING X0 BASED ON A FUNCTION')
        self.activation = nn.ReLU()
        layers = []
        layers.append(nn.Linear(2,128))
        layers.append(self.activation)
        layers.append(nn.Linear(128,1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
    def forward(self,x):
        x0 = self.net(x)
        return x0


class V_elliptical(nn.Module):
    def __init__(self, m, diag_flag, nn_Q,nn_x0):
        super(V_elliptical, self).__init__()

        self.latent_dim = m**2
        
        self.diag_Q = diag_flag
        self.nn_Q = nn_Q
        self.nn_x0 = nn_x0
        if self.nn_Q:
            self.diag_Q = True
        if self.diag_Q:
            print("V_elliptical initialized with a DIAGONAL Q.")
        else:
            print("V_elliptical initialized with a FULL Q.")
        
        if self.nn_Q:
            self.Q_func = Q_func()
        else:
            # lower triangular elements of L (with L^T L = Q)
            self.log_diag_L = nn.Parameter(torch.zeros(self.latent_dim))
            if not self.diag_Q:
                tril_indices = torch.tril_indices(row=self.latent_dim, col=self.latent_dim, offset=-1)
                self.off_diag_L = nn.Parameter(torch.randn(len(tril_indices[0])) * 0.1) # Initialize with small random values

                # We store the indices as a buffer, so they are part of the model's state but not its parameters.
                self.register_buffer('tril_indices', tril_indices)

            self.Q = None  # Placeholder for the symmetric positive-definite matrix Q``
            self.x_0 = nn.Parameter(torch.randn(1, m**2))

        if self.nn_x0:
            self.x0_func = x0_func()
        elif not nn_Q:
            self.x_0 = nn.Parameter(torch.randn(1, m**2))             

    def _construct_x0(self,x=None):
        x0 = self.x0_func(x)
        return x0
    def _construct_Q(self,x=None):
        """
        Constructs the symmetric positive-definite matrix V_elliptical (Q) from L.
        """
        if self.nn_Q:
            Q = self.Q_func(x) 
            
        else: 
            if self.diag_Q:
                Q = torch.exp(2*self.log_diag_L) # shape: m**2
            else:
                # Create an empty n x n matrix for L
                L = torch.zeros(self.latent_dim, self.latent_dim, device=self.log_diag_L.device)

                # Set the diagonal elements using the log_diag_L parameters.
                # The exp() ensures the diagonal is always positive. **** positive diagonal means L is a unique solution to A = LLT. that way we aren't getting the same Q with different L's (redundant, probably confusing during training)
                L.diagonal().copy_(torch.exp(self.log_diag_L))

                # Set the off-diagonal elements from the learned parameters. (ONLY WHEN DIAGONAL IS FALSE)
                L[self.tril_indices[0], self.tril_indices[1]] = self.off_diag_L

                # Compute Q = LLᵀ
                Q = torch.matmul(L, L.T) # shape: m**2 x m**2
        return Q

        
    def forward(self, x):
        if self.nn_Q:
            Q = self._construct_Q(x=x[1])
            Q = Q.reshape(1,self.latent_dim)
        else:
            Q = self._construct_Q()

        if self.nn_x0:
            x0 = self._construct_x0(x=x[1])
            diff = x[0]-x0.reshape(1,self.latent_dim)
        elif self.nn_Q:
                diff = x[0]
        else:
            diff = x[0] - self.x_0

        self.Q = Q
        
        if not self.diag_Q:
            V = torch.einsum('bi,ij,bj->b', diff, Q, diff)
        else:
            V = torch.sum(diff ** 2 * Q, dim=1)
        return V



class ECO(nn.Module):
    def __init__(self,model_params):
        super(ECO,self).__init__()

        m = model_params['m']
        n = model_params['n']
        c0 = model_params['c0']
        project = model_params['project']
        diag_Q = model_params['diag_Q']
        self.nn_Q = model_params['nn_Q']
        self.nn_x0 = model_params['nn_x0']
        dt = model_params['dt']
        self.backbone = model_params['backbone']

        branch_conv_channels = model_params['branch_conv_channels']
        branch_fc_dims = model_params['branch_fc_dims']
        
        # circular_padding = model_params['circular_padding']
        # if circular_padding:
        #     print('Using Circular Padding in Conv Layers')

        trunk_hidden_dims = model_params['trunk_hidden_dims']
        
        # Add a flag for SiLU activation option
        activation_choice = model_params.get('activation', 'ReLU')
        if activation_choice == 'ReLU':
            activation_module = nn.ReLU()
        elif activation_choice == 'SiLU':
            activation_module = nn.SiLU()
        print(f'Using Activation: {activation_choice}')
        
        output_dim = model_params['output_dim']

        print(f'PROJECTION STATUS: {project}')

        ## FOR DEEPONET
        if self.backbone == 'deeponet':
            conv_channels = branch_conv_channels

            # Define the kernel and stride you want to use for all layers
            kernel = 5
            stride = 2

            # Use a list comprehension to build the configuration list
            conv_setup = [
                {'out_channels': channels, 'kernel_size': kernel, 'stride': stride}
                for channels in conv_channels
            ]

            # Create the Branch Net
            self.Branch = Branch(m, conv_config=conv_setup, fc_dims=branch_fc_dims, output_dim=output_dim, activation=activation_module)#, circ_padding=circular_padding)
            self.Trunk = Trunk(n, hidden_dims=trunk_hidden_dims, output_dim=output_dim, activation=activation_module)#, last_act=model_params['trunk_last_act'])

            # Check network structure
            print("--- Initialized Branch Net Structure ---")
            print(self.Branch)
            print("\n--- Initialized Trunk Net Structure ---")
            print(self.Trunk)
            print("-" * 40)
        # FOR FNO
        elif self.backbone == 'fno':
            in_dim = 1
            out_dim = 1
            S = m 
            modes = 20
            width = 128
            self.FNO = Net2d(in_dim, out_dim, S, modes, width)
            print("\n--- Initialized FNO Backbone ---")
            print(self.FNO)

        ## FOR ALL BACKBONES
        self.project = project
        if self.project:
            print('-- Projection is ON --')
        self.c0 = c0
        self.dt = dt
        self.m = m

        self.b = nn.Parameter(torch.tensor(0.0))
        if self.project:            
            self.active_projection_percentage = 0.0
            self.c = nn.Parameter(torch.tensor(self.c0))
            self.c.requires_grad = False
            self.eps_proj = 1e-3
            self.V = V_elliptical(m=m, diag_flag=diag_Q, nn_Q=self.nn_Q, nn_x0=self.nn_x0)

    def discrete_project(self, w_in, w_out, smooth_choice=True, scale_level_set=0.99): 
        V = self.V(w_in)

        b = V + F.relu(-V + self.c ** 2)
        b = scale_level_set * b
        sqrt_b = torch.sqrt(b).unsqueeze(1)
        if self.nn_x0:
            w0 = self.V._construct_x0(x=w_out[1]).reshape(1, self.V.latent_dim)
            w = w_out[0] - w0
        elif self.nn_Q:
            w = w_out[0]
        else:
            w_0 = self.V.x_0
            w = w_out[0] - w_0
        
        V_out = self.V(w_out)
        sqrt_V = torch.sqrt(V_out).unsqueeze(1)

        if self.nn_x0:
            w_proj = w0 + sqrt_b/sqrt_V * (w)
        elif self.nn_Q:
            w_proj = sqrt_b/sqrt_V * (w)
        else:
            w_proj = w_0 + sqrt_b/sqrt_V * (w)
        if smooth_choice:
            k_choice = 100.0
            choice = 1 - torch.sigmoid(k_choice * (V_out - b))
        else:
            choice = (V_out <= b).float()
        choice = choice.reshape(-1, 1)  
        
        active_threshold = 1e-5
        active_proj_count = torch.sum(choice < active_threshold)
        
        batch_size = w_in[0].shape[0]
        self.active_projection_percentage = active_proj_count.item() / batch_size * 100
        
        w_star = choice * w_out[0] + (1 - choice) * w_proj

        return w_star

    def forward(self,x):
        if self.backbone == 'deeponet':
            x1 = self.Branch(x[0])
            x2 = self.Trunk(x[1])
            x_out = torch.einsum("bi,ai->ba",x1,x2)
            x_out += self.b
        elif self.backbone == 'fno':
            x_out = self.FNO(x[0].reshape(-1,self.m,self.m,1))
            x_out = x_out.reshape(-1,self.m*self.m)

        # for any backbone
        if self.project:
            x_out = self.discrete_project(x, (x_out,x[1]))
        return x_out