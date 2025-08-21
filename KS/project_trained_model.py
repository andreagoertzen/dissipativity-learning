import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from model import DeepONet

device = torch.device('cpu')

m = 512
n = 1

model_params = {
    'm': m,
    'n': n,
    'trainable_c': False,
    'c0': 30.0,
    'project': True,
    'diag_Q': True,
    'branch_conv_channels': [32,64,128,256],
    'branch_fc_dims': [256],
    'trunk_hidden_dims': [256,256,256],
    'output_dim': 256,
    'dt': 0.2,
    'discrete_proj': True,
}

model = DeepONet(model_params).to(device)
model_folder = 'Trained_Models/0819_23/E40000_TS0.05_branchConv4_trunkHidden3__dim256'
model.load_state_dict(torch.load(f"./{model_folder}/model_epoch_best.pt", map_location = device))