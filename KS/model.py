import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# class Branch(nn.Module):
#     def __init__(self, m, activation=F.relu):
#         super(Branch, self).__init__()
#         self.m = m
#         self.activation = activation

#         # self.reshape = lambda x: x.view(-1, 1, 28, 28)
#         self.reshape = lambda x: x.view(-1, 1, m)
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2)
#         self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2)
#         self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=2)
#         self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=2)
#         self.flatten = nn.Flatten()
#         # self.fc1 = nn.Linear(128 * 4 * 4, 128) 
#         # self.fc1 = nn.Linear(1280, 256) 
#         # self.fc1 = nn.Linear(1664, 256) 
#         # self.fc1 = nn.Linear(1856, 256) 
#         # self.fc1 = nn.Linear(1984, 256) 
#         self.fc1 = nn.Linear(m, 128)
#         self.fc2 = nn.Linear(128, 128)

#     def forward(self, x):
#         x = self.reshape(x)
#         # x = self.activation(self.conv1(x))
#         # x = self.activation(self.conv2(x))
#         # x = self.activation(self.conv3(x))
#         # x = self.activation(self.conv4(x))
#         x = self.flatten(x)
#         x = self.activation(self.fc1(x))
#         x = self.fc2(x)
#         return x

class Branch(nn.Module):
    def __init__(self, m, conv_config, fc_dims, output_dim=128, activation=nn.ReLU()):
        super(Branch, self).__init__()
        self.activation = activation
        self.reshape = lambda x: x.view(-1, 1, m)

        # --- 1. Build the Convolutional Part Programmatically ---
        conv_layers = []
        in_channels = 1
        for cfg in conv_config:
            conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=cfg['out_channels'],
                    kernel_size=cfg['kernel_size'],
                    stride=cfg['stride']
                )
            )
            conv_layers.append(self.activation)
            in_channels = cfg['out_channels'] # Update for the next layer
        
        self.conv_net = nn.Sequential(*conv_layers)

        # --- 2. Use a Dummy Forward Pass to Find the Flattened Size ---
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, m) # Batch size of 1, 1 channel
            dummy_output = self.conv_net(dummy_input)
            flattened_size = dummy_output.flatten(1).shape[1]
            print(f"Auto-detected flattened size for FC layer: {flattened_size}")

        # --- 3. Build the Fully-Connected Part Programmatically ---
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
    def __init__(self, n, hidden_dims, output_dim=128, activation=nn.ReLU()):
        super(Trunk, self).__init__()
        self.activation = activation
        
        # Create a list of all layer dimensions
        all_dims = [n] + hidden_dims + [output_dim]
        
        layers = []
        for i in range(len(all_dims) - 1):
            layers.append(nn.Linear(all_dims[i], all_dims[i+1]))
            # Add activation to all layers except the last one
            if i < len(all_dims) - 2:
                layers.append(self.activation)
        
        # Create the sequential model
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class V_elliptical(nn.Module):
    def __init__(self, m, diag_flag):
        super(V_elliptical, self).__init__()

        self.latent_dim = m
        
        self.diag_Q = diag_flag
        if self.diag_Q:
            print("V_elliptical initialized with a DIAGONAL Q.")
        else:
            print("V_elliptical initialized with a FULL Q.")
        
        # diagonal elements of the lower triangular matrix L
        self.log_diag_L = nn.Parameter(torch.zeros(self.latent_dim))

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

        if not self.diag_Q:
            # Set the off-diagonal elements from the learned parameters. (ONLY WHEN DIAGONAL IS FALSE)
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
    def __init__(self,model_params):
        super(DeepONet,self).__init__()

        m = model_params['m']
        n = model_params['n']
        trainable_c = model_params['trainable_c']
        c0 = model_params['c0']
        project = model_params['project']
        diag_Q = model_params['diag_Q']
        dt = model_params['dt']

        branch_conv_channels = model_params['branch_conv_channels']
        branch_fc_dims = model_params['branch_fc_dims']
        
        trunk_hidden_dims = model_params['trunk_hidden_dims']
        
        output_dim = model_params['output_dim']

        # Define a configuration for the convolutional layers
        # Define the desired output channels for each convolutional layer
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
        self.Branch = Branch(m, conv_config=conv_setup, fc_dims=branch_fc_dims, output_dim=output_dim)
        self.Trunk = Trunk(n, hidden_dims=trunk_hidden_dims, output_dim=output_dim)

        # Check network structure (for debugging)
        print("--- Initialized Branch Net Structure ---")
        print(self.Branch)
        print("\n--- Initialized Trunk Net Structure ---")
        print(self.Trunk)
        print("-" * 40)
        
        self.project = project
        self.c0 = c0
        self.dt = dt

        self.trainable_c = trainable_c

        self.b = nn.Parameter(torch.tensor(0.0))
        if self.project:
            print('Projection layer included')
            
            self.c = nn.Parameter(torch.tensor(self.c0))
            if self.trainable_c:
                # freeze self.c gradient
                self.c.requires_grad = True
            else:
                self.c.requires_grad = False
            self.eps_proj = 1e-3
            self.V = V_elliptical(m=m, diag_flag=diag_Q)


    def f_project(self,w_in,w_out,dt):
        w0 = self.V.x_0
        V = self.V(w_in)
        Q = self.V.Q
        diff = w_in-w0
        dVdw = torch.einsum('ij,bj->bi',2*Q,diff)
        # dVdw = 2 * (diff @ Q)  # Gradient of V with respect to w_in

        # constraint has the form Ay + b(x) <= 0
        A = dVdw
        # bx = V-(1/dt) * torch.einsum('bi,bi->b',dVdw, w_in) - self.c**2
        # w_star = w_out - dVdw * (F.relu( torch.einsum('bi,bi->b',A,w_out) + bx)/torch.clamp((dVdw**2).sum(dim=1), min=self.eps_proj)).unsqueeze(1)
        bx = - (A * w_in).sum(dim=1) + dt * (V - self.c ** 2)

        # print((F.relu(A * w_out).sum(dim=1)).unsqueeze(1).shape, A.shape, bx.shape)

        w_star = w_out - A * (F.relu( (A * w_out).sum(dim=1) + bx) ).unsqueeze(1) / torch.clamp((dVdw ** 2).sum(dim=1), min=self.eps_proj).unsqueeze(1)

        return w_star


    def forward(self,x):
        x1 = self.Branch(x[0])
        x2 = self.Trunk(x[1])
        x_out = torch.einsum("bi,ai->ba",x1,x2)
        x_out += self.b
        if self.project:
            x_out = self.f_project(x[0],x_out, dt=self.dt)
        return x_out