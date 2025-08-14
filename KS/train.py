import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import jax
import model
from model import run_model_visualization
import os


device = torch.device('cuda')
torch.manual_seed(0)
iterations = 2000000
# iterations = 200000
n_save = 1000
bsize = 10000
lam_reg_vol = 0.01
project = False
tag = 'test'
model_folder = f'models_{tag}'
figs_folder = f'figs_{tag}'


if not os.path.exists(model_folder):
    os.makedirs(model_folder)
if not os.path.exists(figs_folder):
    os.makedirs(figs_folder)

data = np.load('KS_data_l64.0.npz',allow_pickle=True)
physical_data = data['Physical']
physical_u = np.array(physical_data.item()['u'])
physical_x = np.array(physical_data.item()['x'])
physical_t = np.array(physical_data.item()['t'])


def get_data(u,x):
    u = u[0::10,...]
    x_data = (torch.tensor(u[:-1,...]).to(device),torch.tensor(x.reshape(len(x),1)).to(device))
    y_data = torch.tensor(u[1:,...]).to(device)

    return x_data,y_data

def get_batch(x,y,bsize):
    ind_time = np.random.randint(0,x[0].shape[0],bsize)
    x_out1 = x[0][ind_time,:]
    x_batch = (x_out1.to(device),x[1].to(device))
    y_batch = y[ind_time,:].to(device)
    return x_batch,y_batch

def ellip_vol(model):
    # L = torch.tril(model.V.L)
    # L.diagonal().exp_()
    # Q = L @ L.t()
    Q = model.V._construct_Q()

    det_Q = torch.linalg.det(Q)
    m = model.Branch.m
    # n = model.state_dim
    # Correct implementation
    # vol = torch.sqrt((model.c**2)**n / det_Q)
    # Legacy implementation
    vol = torch.sqrt((model.c**2) / det_Q)
    # we omit the gamma function as it is a constant for the same model
    vol = np.pi ** (m/2) * vol

    return vol

x_test,y_test = get_data(physical_u[90000:,...],physical_x)
x_train,y_train = get_data(physical_u[:90000,...],physical_x)
print(y_test.shape)
print(y_train.shape)

m = s = physical_u.shape[1]
n = x_train[1].shape[1]
print(n)
model = model.DeepONet(m,n,project = project).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)
num_params = sum(v.numel() for v in model.parameters() if v.requires_grad)
print(f'model params: {num_params}')

loss_func = torch.nn.MSELoss()
tic = time.time()

best_loss = 1e5
model.train()

losses = []
test_losses = []

for j in range(iterations+1):

    x_batch,y_batch = get_batch(x_train,y_train,bsize=bsize)
    optimizer.zero_grad() 
    u_pred2 = model(x_batch)#.unsqueeze(1)
    dynamic_loss = loss_func(u_pred2,y_batch)

    if project:
        vol = ellip_vol(model)
        reg_loss = lam_reg_vol * vol.squeeze()
        # reg_loss = torch.tensor(0.0, device=device)
    else:
        reg_loss = torch.tensor(0.0, device=device)
    
    loss = dynamic_loss + reg_loss

    loss.backward()
    optimizer.step()

    if j%n_save == 0:
        toc = time.time()
        total_time = (toc-tic)
        u_test = model(x_test)#.unsqueeze(1)
        loss = loss.to("cpu").detach().numpy()
        loss_test = loss_func(u_test,y_test)
        loss_test = loss_test.to("cpu").detach().numpy()
        # loss_metric = my_loss(u_test,y_test).to("cpu").detach().numpy()
        print(f"iteration: {j}      train loss: {loss:.2e}      test loss: {loss_test:.2e}")#      test metric: {loss_metric:.2e}")
        print(total_time)
        if project:
            print(vol)


        losses.append(loss)
        test_losses.append(loss_test)
        plt.figure(figsize=(6, 4))
        plt.plot(np.linspace(0,j,int(j/n_save+1)),losses, label='Training Loss')
        plt.plot(np.linspace(0,j,int(j/n_save+1)),test_losses, label='Test Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss (log scaled)')
        plt.yscale('log')
        plt.title('Loss over Time')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"figs_{tag}/loss_iter.png")
        plt.close()

        if loss_test < best_loss:
            torch.save(model.state_dict(),f"./models_{tag}/model_step" + str(j))
            best_ind = j
            best_loss = loss_test
        tic = time.time()
    if j%200000 == 0:
        model.eval()
        run_model_visualization(model,x_test,y_test,s,device,figs_dir = tag)

print(best_ind)

# best_ind = 93000
## inference
model.load_state_dict(torch.load(f'models_{tag}/model_step'+str(best_ind)))
model.eval()
run_model_visualization(model,x_test,y_test,s,device,figs_dir = tag)


