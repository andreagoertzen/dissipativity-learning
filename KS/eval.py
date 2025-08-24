import numpy as np 
import torch
from model import DeepONet
import matplotlib.pyplot as plt
from utils import visualize_ellipsoid, compare_distributions, rollout_on_test, pca_histogram_eval, evaluate_fourier_spectrum, run_model_visualization


### LOAD MODEL
print('LOADING MODEL')
m = s = 512
n = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_params = {
    'm': m,
    'n': n,
    'trainable_c': False,
    'c0': 0.0, # doesn't matter, we will load the correct value
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

model_folder = 'Trained_Models/0822_15/E40000_TS0.05_branchConv3_trunkHidden3_dt1.0__proj_LamRegVol0.1_C0100.0_diagQdiscreteProj_dim256_train2'
figs_dir = 'Trained_Models/0822_15/E40000_TS0.05_branchConv3_trunkHidden3_dt1.0__proj_LamRegVol0.1_C0100.0_diagQdiscreteProj_dim256_train2/eval_results'
model.load_state_dict(torch.load(f'{model_folder}/model_epoch_best.pt',map_location=device))
model.eval()

## GET MODEL PARAMETERS
if model.project:
    Q = model.V._construct_Q().detach().cpu().numpy()
    c = model.c.detach().cpu().numpy() ** 2
else:
    Q = None
    c = 30.0 ** 2

### LOAD DATA
print('LOADING TEST DATA')
trunk_scale = 0.05
file_dir = 'Data/KS_data_test_l100.53_grid512_M1_T2000.0_dt0.005_dt_sample0.2_amp20.0.npz/data.npz'
data = np.load(file_dir, allow_pickle=True)
x = torch.tensor(data['x'],dtype=torch.float32).to(device)
x = x.reshape(s,1)
gt_traj = data['u_batch'][:,::5,:]
print(gt_traj.shape)

## ONE STEP ON TEST DATA AND ROLLOUT ON RANDOM IC
print('ONE STEP AND RANDOM IC VISUALS')
run_model_visualization(
    model = model,
    x_test=(torch.tensor(gt_traj[0,:999,:],dtype=torch.float32).to(device),x*trunk_scale),
    y_test=torch.tensor(gt_traj[0,1:1000,:],dtype=torch.float32).to(device),
    s=s,
    device=device,
    figs_dir=figs_dir,
    figs_tag = '',
    rollout_steps_test=2000,
    rollout_steps_random=10000,
    random_seed=10,
    random_IC_mag=5.0
    )

## ROLLOUT TRAJECTORY AND V PLOT
print('ROLLOUT TRAJECTORY')
pred_traj = rollout_on_test(model, 
    data_x=x, 
    trunk_scale=trunk_scale, 
    test_traj=gt_traj, 
    device=device, 
    figs_dir=figs_dir, 
    project=model.project,
    c=c
    )
print(pred_traj.shape)

## PCA PLOT
# pred_traj_pca = pred_traj[pred_traj<1e6]

row_norms = torch.norm(pred_traj, p=2, dim=2)
mask = row_norms<1e6
print(mask.shape)
pred_traj_pca = pred_traj[mask,:]

print('PCA PROJECTION')
print(pred_traj.shape)
print(pred_traj_pca.shape)
pca_traj_gt, pca_traj_pred = visualize_ellipsoid(gt_traj = gt_traj[0,...], 
    test_traj = pred_traj_pca, 
    figs_dir=figs_dir, 
    Q=Q, 
    c=c,
    tag='')

## DISTRIBUTION COMPARISON FOR DATA
print('DISTRIBUTION COMPARISON FOR TRAJECTORY')
pred_traj_np = pred_traj[0,...].detach().cpu().numpy()
kl_div_traj = compare_distributions(gt_traj = gt_traj[0,...].ravel(), 
    pred_traj = pred_traj_np.ravel(), 
    bins = 50,
    plot=True, 
    save_name=f'{figs_dir}/distribution_traj.png')


# ## DISTRIBUTION COMPARISON FOR PCA MODES
print('DISTRIBUTION COMPARISON FOR PCA MODES')
pca_histogram_eval(gt_pca=pca_traj_gt, 
    pred_pca=pca_traj_pred, 
    bins=50, 
    lim=[[-50.0, 50.0], [-50.0, 50.0]], 
    save_path=f'{figs_dir}/distribution_pca.png', 
    title_gt='Ground Truth', 
    title_pred='Prediction')

## FOURIER SPECTRUM
print('FOURIER SPECTRUM COMPARISON')
print(gt_traj.shape)
print(pred_traj.shape)
final_error_star = evaluate_fourier_spectrum(gt_traj = gt_traj[0,...], 
    star_traj=pred_traj_np, 
    save_path=f'{figs_dir}/fourier_spectrum.png')


## PLOT GT AND PREDICTED TRAJ
aspect = 1/2*pred_traj.shape[1]/pred_traj.shape[2]
plt.figure()
plt.imshow(
    pred_traj[0,...].T.detach().cpu().numpy().astype(np.float32),
    # extent=[0, rollout_steps_random, 0, s],
    vmin = -5, vmax = 5,
    aspect=aspect
)
plt.title('Rollout from Validation Initial Condition')
plt.colorbar()
plt.xlabel('Time (s)')
plt.ylabel('Position')
plt.savefig(f'{figs_dir}/rollout.png')
plt.close()




# model.load_state_dict(torch.load(f'{model_folder}/model_epoch_best.pt',map_location=device))
# model.eval()

# ## GET MODEL PARAMETERS
# if model.project:
#     Q = model.V._construct_Q().detach().cpu().numpy()
#     c = model.c.detach().cpu().numpy() ** 2
# else:
#     Q = None
#     c = 30.0 ** 2

# ### LOAD DATA
# print('LOADING TEST DATA')
# trunk_scale = 0.05
# file_dir = 'Data/KS_data_test_l100.53_grid512_M1_T2000.0_dt0.005_dt_sample0.2_amp20.0.npz/data.npz'
# data = np.load(file_dir, allow_pickle=True)
# x = torch.tensor(data['x'],dtype=torch.float32)
# x = x.reshape(s,1)
# gt_traj = data['u_batch'][:,::5,:]
# print(gt_traj.shape)

# ## ONE STEP ON TEST DATA AND ROLLOUT ON RANDOM IC
# print('ONE STEP AND RANDOM IC VISUALS')
# run_model_visualization(
#     model = model,
#     x_test=(torch.tensor(gt_traj[0,:999,:],dtype=torch.float32),x*trunk_scale),
#     y_test=torch.tensor(gt_traj[0,1:1000,:],dtype=torch.float32),
#     s=s,
#     device=device,
#     figs_dir=figs_dir,
#     figs_tag = '',
#     rollout_steps_test=2000,
#     rollout_steps_random=10000,
#     random_seed=10,
#     random_IC_mag=5.0
#     )

# ## ROLLOUT TRAJECTORY AND V PLOT
# print('ROLLOUT TRAJECTORY')
# pred_traj = rollout_on_test(model, 
#     data_x=x, 
#     trunk_scale=trunk_scale, 
#     test_traj=gt_traj, 
#     device=device, 
#     figs_dir=figs_dir, 
#     project=model.project,
#     c=c
#     )
# print(pred_traj.shape)

# ## PCA PLOT
# print('PCA PROJECTION')
# pca_traj_gt, pca_traj_pred = visualize_ellipsoid(gt_traj = gt_traj[0,...], 
#     test_traj = pred_traj[0,...], 
#     figs_dir=figs_dir, 
#     Q=Q, 
#     c=c,
#     tag='')

# ## DISTRIBUTION COMPARISON FOR DATA
# print('DISTRIBUTION COMPARISON FOR TRAJECTORY')
# kl_div_traj = compare_distributions(gt_traj = gt_traj[0,...].ravel(), 
#     pred_traj = pred_traj[0,...].ravel(), 
#     bins = 50,
#     plot=True, 
#     save_name=f'{figs_dir}/distribution_traj.png')


# # ## DISTRIBUTION COMPARISON FOR PCA MODES
# print('DISTRIBUTION COMPARISON FOR PCA MODES')
# pca_histogram_eval(gt_pca=pca_traj_gt, 
#     pred_pca=pca_traj_pred, 
#     bins=50, 
#     lim=[[-50.0, 50.0], [-50.0, 50.0]], 
#     save_path=f'{figs_dir}/distribution_pca.png', 
#     title_gt='Ground Truth', 
#     title_pred='Prediction')

# ## FOURIER SPECTRUM
# print('FOURIER SPECTRUM COMPARISON')
# print(gt_traj.shape)
# print(pred_traj.shape)
# final_error_star = evaluate_fourier_spectrum(gt_traj = gt_traj[0,...], 
#     star_traj=pred_traj[0,...], 
#     save_path=f'{figs_dir}/fourier_spectrum.png')


# ## PLOT GT AND PREDICTED TRAJ
# aspect = 1/2*pred_traj.shape[1]/pred_traj.shape[2]
# plt.figure()
# plt.imshow(
#     pred_traj[0,...].T.detach().numpy().astype(np.float32),
#     #extent=[0, rollout_steps_random, 0, s],
#     vmin = -5, vmax = 5,
#     aspect=aspect
# )
# plt.title('Rollout from Validation Initial Condition')
# plt.colorbar()
# plt.xlabel('Time (s)')
# plt.ylabel('Position')
# plt.savefig(f'{figs_dir}/rollout.png')
# plt.close()