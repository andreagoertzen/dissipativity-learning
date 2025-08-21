import numpy as np 
import torch
from model import DeepONet
import matplotlib.pyplot as plt

m = s = 512
n = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#### LOAD MODEL
model_params = {
    'm': m,
    'n': n,
    'trainable_c': False,
    'c0': 45.0,
    'project': True,
    'diag_Q': True,
    'branch_conv_channels': [32,64,128,256],
    'branch_fc_dims': [256],
    'trunk_hidden_dims': [256,256,256],
    'output_dim': 256,
    'dt': 1,
    'discrete_proj': True,
}

print('initializing model')
model = DeepONet(model_params).to(device)
model_file = 'Trained_Models/0821_03/E40000_TS0.05_branchConv4_trunkHidden3_dt1.0__proj_LamRegVol0.1_C045.0_diagQdiscreteProjwarmStart_dim256_train500'
# model_file = 'Trained_Models/0821_03/E40000_TS0.05_branchConv4_trunkHidden3_dt1.0__dim256_train500'

print('loading saved model')
model.load_state_dict(torch.load(f'{model_file}/model_epoch_best.pt',map_location=device))
model.eval()

##### LOAD DATA
trunk_scale = 0.05
file_dir = 'Data/KS_data_batched_l100.53_grid512_M8_T500.0_dt0.01_amp5.0/data.npz'
data = np.load(file_dir, allow_pickle=True)
x = torch.tensor(data['x'],dtype=torch.float32)
x = x.reshape(s,1)*trunk_scale



figs_dir = 'Trained_Models/0821_03/E40000_TS0.05_branchConv4_trunkHidden3_dt1.0__proj_LamRegVol0.1_C045.0_diagQdiscreteProjwarmStart_dim256_train500/eval_results'
# figs_dir = 'Trained_Models/0821_03/E40000_TS0.05_branchConv4_trunkHidden3_dt1.0__dim256_train500/eval_results'


## ROLLOUT
print('rollout')
random_seed = 10
rollout_steps_random = 1000

torch.manual_seed(random_seed)
u0 = torch.randn(1, s).to(device)*5

print("Random IC shape:", u0.shape)
rollout_traj = torch.zeros(rollout_steps_random, s)
V = torch.zeros(rollout_steps_random,1)
u_out = u0

Q = model.V._construct_Q()

with torch.no_grad():
    for i in range(rollout_steps_random):
        u_out = model((u_out, x))
        rollout_traj[i, :] = u_out
        V[i] = u_out @ Q @ u_out.T

plt.figure()
plt.imshow(
    rollout_traj.T.detach().numpy().astype(np.float32),
    extent=[0, rollout_steps_random, 0, s]#,
    #aspect='auto'
)
plt.title('Rollout from Random Initial Condition')
plt.colorbar()
plt.xlabel('Time')
plt.ylabel('Position')
plt.savefig(f'{figs_dir}/rollout_randomIC_1000.png')
plt.close()

plt.figure()
plt.plot(V)
plt.xlabel('time')
plt.ylabel('V')
plt.savefig(f'{figs_dir}/V_plot_randomIC_1000.png')
plt.close()





# file_dir = 'Data/KS_data_batched_l100.53_grid512_M8_T500.0_dt0.01_amp5.0/data.npz'
# data = np.load(file_dir, allow_pickle=True)
# data = dict(data)
# print('CHECKING DATA SHAPE')
# print(data['u_batch'].shape)
# data['u_batch'] = data['u_batch'][:,::100,:]
# print(data['u_batch'].shape) 
# print(data['t'])

