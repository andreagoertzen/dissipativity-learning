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
    'c0': 0.0,
    'project': True,
    'diag_Q': True,
    'branch_conv_channels': [32,64,128],
    'branch_fc_dims': [256],
    'trunk_hidden_dims': [256,256,256],
    'output_dim': 256,
    'dt': 1,
    'discrete_proj': True,
}

print('initializing model')
model = DeepONet(model_params).to(device)
# model_folder = 'Trained_Models/0821_03/E40000_TS0.05_branchConv4_trunkHidden3_dt1.0__proj_LamRegVol0.1_C060.0_diagQdiscreteProjwarmStart_dim256_train500'
model_folder = 'Trained_Models/0822_15/E40000_TS0.05_branchConv3_trunkHidden3_dt1.0__proj_LamRegVol0.1_C060.0_diagQdiscreteProj_dim256_train2'
# model_folder = 'Trained_Models/0821_03/E40000_TS0.05_branchConv4_trunkHidden3_dt1.0__dim256_train500'

print('loading saved model')
model.load_state_dict(torch.load(f'{model_folder}/model_epoch_best.pt',map_location=device))
model.eval()

##### LOAD DATA
trunk_scale = 0.05
file_dir = 'Data/KS_data_test_l100.53_grid512_M1_T2000.0_dt0.005_dt_sample0.2_amp20.0.npz/data.npz'
data = np.load(file_dir, allow_pickle=True)
x = torch.tensor(data['x'],dtype=torch.float32).to(device)
x = x.reshape(s,1)*trunk_scale
gt_traj = data['u_batch'][:,::5,:]
print(gt_traj.shape)



# figs_dir = 'Trained_Models/0821_03/E40000_TS0.05_branchConv4_trunkHidden3_dt1.0__proj_LamRegVol0.1_C060.0_diagQdiscreteProjwarmStart_dim256_train500/eval_results'
# figs_dir = 'Trained_Models/0821_05/E40000_TS0.05_branchConv4_trunkHidden3_dt1.0__proj_LamRegVol0.1_C045.0_diagQdiscreteProj_dim256_train500/eval_results'
# figs_dir = 'Trained_Models/0821_03/E40000_TS0.05_branchConv4_trunkHidden3_dt1.0__dim256_train500/eval_results'
save_dir = model_folder + '/eval_results'

## ROLLOUT
print('rollout')
random_seed = 10
rollout_steps_random = 2000
rollout_steps_random = 2000

torch.manual_seed(random_seed)

mags = [1,5,10,20,100]
u0 = torch.zeros(len(mags),s)

for i,mag in enumerate(mags):
    u0[i,:] = torch.randn(1, s).to(device)*mag

print("Random IC shape:", u0.shape)
# rollout_traj = torch.zeros(len(mags), rollout_steps_random, s)
# V = torch.zeros(len(mags),rollout_steps_random)
# V_gt = torch.zeros(1,rollout_steps_random)
# u_out = u0
# Q = model.V._construct_Q().detach().cpu().numpy()
# c = model.c.detach().cpu().numpy() ** 2
# gt_traj = torch.tensor(gt_traj,dtype=torch.float32).to(device)

rollout_traj = torch.zeros(1, rollout_steps_random, s).to(device)
u_out = torch.tensor(gt_traj[:,0,:],dtype=torch.float32).to(device)
rollout_traj[:,0,:] = u_out
with torch.no_grad():
    for i in range(1,rollout_steps_random):
        # V[:,i] = model.V(u_out)
        # V_gt[0,i] = gt_traj[:,i,:] @ Q @ gt_traj[:,i,:].T
        u_out = model((u_out, x))
        rollout_traj[:,i, :] = u_out
        # V[:,i] = u_out @ Q @ u_out.T

# rollout_dict = {
#     "mags": mags,
#     "rollout_traj": np.array(rollout_traj)
#     }
plt.imshow(rollout_traj[0,...].T.detach().cpu().numpy().astype(np.float32))
plt.savefig(f'{save_dir}/test.png')
# np.savez(f'{save_dir}/rollout_data.npz',rollout_traj.detach().cpu().numpy())
np.savez(f'{save_dir}/rollout_data2.npz',rollout_traj)


# aspect = 1/2*rollout_traj.shape[1]/rollout_traj.shape[2]
# for i,mag in enumerate(mags):
#     plt.figure()
#     plt.imshow(
#         rollout_traj[i,...].T.detach().numpy().astype(np.float32),
#         extent=[0, rollout_steps_random, 0, s],
#         vmin = -5, vmax = 5,
#         aspect=aspect
#     )
#     plt.title('Rollout from Random Initial Condition')
#     plt.colorbar()
#     plt.xlabel('Time (s)')
#     plt.ylabel('Position')
#     plt.savefig(f'{figs_dir}/rollout_randomIC_{rollout_steps_random}_mag{mag}.png')
#     plt.close()

#     plt.figure()
#     plt.plot(V[i,...],label = 'Prediction')
#     plt.plot(V_gt[0,...],label='Ground Truth')
#     plt.xlabel('time')
#     plt.ylabel('V')
#     plt.yscale('log')
#     plt.legend()
#     plt.savefig(f'{figs_dir}/V_plot_randomIC_{rollout_steps_random}_mag{mag}.png')
#     plt.close()

#     print('starting PCA')

#     kl_div = compare_distributions(gt_traj[0,...].ravel(), rollout_traj[i,...].ravel(), plot=True, save_name=f'{figs_dir}/distribution_mag{mag}.png')
#     visualize_ellipsoid(gt_traj, rollout_traj[i,...], figs_dir, Q=Q, c=c,tag=f'mag{mag}')

