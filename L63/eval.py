#!/usr/bin/env python3
# eval.py

import os, argparse, logging
import numpy as np
import torch, torch.nn as nn
import matplotlib.pyplot as plt
from model import ProjectedMLP
from utils import TrajectoryTensorDataset, gen_real_multi_traj


class OneStepFromSubtraj(torch.utils.data.Dataset):
    def __init__(self, trajectories, stride=1):
        self.base = TrajectoryTensorDataset(trajectories, subtraj_length=2, stride=stride)
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        subtraj = self.base[idx]
        if isinstance(subtraj, torch.Tensor): subtraj = subtraj.numpy()
        return torch.from_numpy(subtraj[0].astype(np.float32)), torch.from_numpy(subtraj[1].astype(np.float32))


def split_by_trajectory(X_ds, val_frac=0.1, seed=0):
    N = X_ds.shape[0]
    rng = np.random.default_rng(seed); perm = rng.permutation(N)
    n_val = max(1,int(N*val_frac))
    return X_ds[perm[n_val:]], X_ds[perm[:n_val]]


@torch.no_grad()
def eval_one_step(val_loader, model, device):
    loss_fn = nn.MSELoss(); val_mse,n=0.0,0
    for xb,yb in val_loader:
        xb,yb=xb.to(device), yb.to(device)
        val_mse += loss_fn(model(xb), yb).item(); n+=1
    return val_mse/max(1,n)


@torch.no_grad()
def closed_loop_rollout(model, x0_np, steps, device):
    x=torch.from_numpy(x0_np.astype(np.float32)).unsqueeze(0).to(device)
    preds=[x.cpu().numpy()[0]]
    for _ in range(steps):
        x=model(x); preds.append(x.detach().cpu().numpy()[0])
    return np.stack(preds,0)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-path',required=True)
    p.add_argument('--model-dir',required=True)
    p.add_argument('--val-frac',type=float,default=0.1)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--ckpt',default='model_epoch_best.pt')
    p.add_argument('--eval-steps',type=int,default=1000)
    p.add_argument('--gt-traj-num',type=int,default=4)
    p.add_argument('--gt-dt',type=float,default=0.01)
    p.add_argument('--gt-seed',type=int,default=123)
    args=p.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_ds=np.load(args.data_path,allow_pickle=True)['X_ds'] if args.data_path.endswith('.npz') else np.load(args.data_path,allow_pickle=True)
    _,T,D=X_ds.shape
    params_npz=np.load(os.path.join(args.model_dir,'model_params.npz'),allow_pickle=True)
    model_cfg={k:params_npz[k].item() if params_npz[k].size==1 else params_npz[k].tolist()
               for k in params_npz.files if k in ['d','hidden_dims','activation','discrete_proj','c0','trainable_c','diag_Q','dt']}
    model_cfg['activation']=nn.GELU() if params_npz['activation']=='gelu' else nn.ReLU()
    model=ProjectedMLP(model_cfg).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir,args.ckpt),map_location=device))

    _,X_val=split_by_trajectory(X_ds,args.val_frac,args.seed)
    val_loader=torch.utils.data.DataLoader(OneStepFromSubtraj(X_val),batch_size=4096,shuffle=False)
    one_step_val_mse=eval_one_step(val_loader,model,device)

    steps=min(args.eval_steps,T-1)
    val_traj=X_val[0]; pred_val=closed_loop_rollout(model,val_traj[0],steps,device)
    rollout_val_mse=float(np.mean((pred_val-val_traj[:steps+1])**2))

    plt.figure(); [plt.plot(val_traj[:steps+1,i],label=f'gt{i}') or plt.plot(pred_val[:,i],'--',label=f'pred{i}') for i in range(min(3,D))]
    plt.legend(); plt.savefig(os.path.join(args.model_dir,'eval_rollout_val_traj.png')); plt.close()

    X_gt=gen_real_multi_traj(M=args.gt_traj_num,N=steps,dt=args.gt_dt,dt_target=0.01)
    if isinstance(X_gt,dict) and "X_ds" in X_gt: X_gt=X_gt['X_ds']
    mse_list=[np.mean((closed_loop_rollout(model,tr[0],steps,device)-tr)**2) for tr in X_gt]
    rollout_gt_mse=float(np.mean(mse_list))

    print(f"One-step Val MSE: {one_step_val_mse:.6e}")
    print(f"Rollout Val MSE:  {rollout_val_mse:.6e}")
    print(f"Rollout GT MSE:   {rollout_gt_mse:.6e}")


if __name__=="__main__":
    logging.basicConfig(level=logging.INFO); main()
