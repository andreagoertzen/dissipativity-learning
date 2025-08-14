import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import model
import os
from utils import TrajectoryDataset, load_multi_traj_data
from torch.utils.data import DataLoader
from tqdm import tqdm

def ellip_vol(model):
    d = model.V.log_diag_L.numel()
    c_val = model.c ** 2
    
    # Compute det(Q)^(-1/2)
    log_det_Q = 2 * torch.sum(model.V.log_diag_L)
    det_factor = torch.exp(- 1/2 * log_det_Q)

    # Final volume
    if model.fix_c_flag:
        vol = det_factor
    else:
        vol = (c_val**(d/2)) * det_factor
    return vol

device = torch.device('cuda')
torch.manual_seed(0)
np.random.seed(0)

epochs = 10000
n_save_epochs = 50
bsize = 2048
lam_reg_vol = 1.0
project = True
tag = 'final_dataloader_run'
model_folder = f'models_{tag}'
figs_folder = f'figs_{tag}'


if not os.path.exists(model_folder):
    os.makedirs(model_folder)
if not os.path.exists(figs_folder):
    os.makedirs(figs_folder)

file_dir = 'Data/KS_data_batched_l100.53_grid512_M8_T500.0_dt0.01_amp5.0/data.npz'
data = np.load(file_dir,allow_pickle=True)

train_dataset, val_dataset = load_multi_traj_data(data)

train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True, pin_memory=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=bsize, shuffle=False)
print(f"Created DataLoaders with {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

# Model Optimizer Initialization
m = s = data['u_batch'].shape[2]  # Assuming u_batch is of shape (num_traj, traj_length, traj_dim)
n = 1
model = model.DeepONet(m,n,project = project).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)
num_params = sum(v.numel() for v in model.parameters() if v.requires_grad)
print(f'model params: {num_params}')

loss_func = torch.nn.MSELoss()
tic = time.time()

best_loss = float('inf')
train_losses = []
val_losses = []
dynamic_losses = []
reg_losses = []

# --- Main Training Loop ---
for epoch in tqdm(range(epochs + 1)):
    model.train()
    epoch_train_loss = 0
    epoch_dynamic_loss = 0
    epoch_reg_loss = 0
    # Iterate over batches from the DataLoader
    for x_batch, y_batch in train_loader:
        # The dataloader gives us a tuple for x_batch
        branch_batch, trunk_batch = x_batch
        
        # Move batch to the correct device
        branch_batch = branch_batch.to(device)
        trunk_batch = trunk_batch.to(device)
        # Transpose the trunk input
        
        trunk_input = trunk_batch[0]
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        
        # The model expects a tuple of (branch_input, trunk_input)
        u_pred = model((branch_batch, trunk_input))
        dynamic_loss = loss_func(u_pred, y_batch)

        # Calculate regularization loss if projection is enabled
        if project:
            vol = ellip_vol(model)
            reg_loss = lam_reg_vol * vol.squeeze()
        else:
            reg_loss = torch.tensor(0.0, device=device)
        
        loss = dynamic_loss + reg_loss
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
        epoch_dynamic_loss += dynamic_loss.item()
        epoch_reg_loss += reg_loss.item()

    # --- Evaluation, Logging, and Checkpointing ---
    if epoch % n_save_epochs == 0:
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                branch_val, trunk_val = x_val
                branch_val, trunk_val = branch_val.to(device), trunk_val.to(device)
                trunk_val_input = trunk_val[0]
                y_val = y_val.to(device)

                u_val_pred = model((branch_val, trunk_val_input))
                epoch_val_loss += loss_func(u_val_pred, y_val).item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_dynamic_loss = epoch_dynamic_loss / len(train_loader)
        avg_reg_loss = epoch_reg_loss / len(train_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        dynamic_losses.append(avg_dynamic_loss)
        reg_losses.append(avg_reg_loss)

        total_time = time.time() - tic

        print(f"Epoch: {epoch}/{epochs} | Train Loss: {avg_train_loss:.3e} | Dynamic Loss: {avg_dynamic_loss:.3e} | Regularization Loss: {avg_reg_loss:.3e} | Val Loss: {avg_val_loss:.3e} | Time: {total_time:.2f}s")

        # Save the model if it has the best validation loss so far
        if avg_val_loss < best_loss:
            print(f"New best model found at epoch {epoch} with validation loss {avg_val_loss:.3e}. Saving...")
            torch.save(model.state_dict(), f"./{model_folder}/model_epoch_best.pt")
            best_loss = avg_val_loss
            best_ind = epoch
        
        tic = time.time()

# for j in range(iterations+1):

#     x_batch,y_batch = get_batch(x_train,y_train,bsize=bsize)
#     optimizer.zero_grad() 
#     u_pred2 = model(x_batch)#.unsqueeze(1)
#     dynamic_loss = loss_func(u_pred2,y_batch)

#     if project:
#         vol = ellip_vol(model)
#         reg_loss = lam_reg_vol * vol.squeeze()
#         # reg_loss = torch.tensor(0.0, device=device)
#     else:
#         reg_loss = torch.tensor(0.0, device=device)
    
#     loss = dynamic_loss + reg_loss

#     loss.backward()
#     optimizer.step()

#     if j%n_save == 0:
#         toc = time.time()
#         total_time = (toc-tic)
#         u_test = model(x_test)#.unsqueeze(1)
#         loss = loss.to("cpu").detach().numpy()
#         loss_test = loss_func(u_test,y_test)
#         loss_test = loss_test.to("cpu").detach().numpy()
#         # loss_metric = my_loss(u_test,y_test).to("cpu").detach().numpy()
#         print(f"iteration: {j}      train loss: {loss:.2e}      test loss: {loss_test:.2e}")#      test metric: {loss_metric:.2e}")
#         print(total_time)
#         if project:
#             print(vol)


#         losses.append(loss)
#         test_losses.append(loss_test)
#         plt.figure(figsize=(6, 4))
#         plt.plot(np.linspace(0,j,int(j/n_save+1)),losses, label='Training Loss')
#         plt.plot(np.linspace(0,j,int(j/n_save+1)),test_losses, label='Test Loss')
#         plt.xlabel('Iteration')
#         plt.ylabel('Loss (log scaled)')
#         plt.yscale('log')
#         plt.title('Loss over Time')
#         plt.grid(True)
#         plt.legend()
#         plt.savefig(f"figs_{tag}/loss_iter.png")
#         plt.close()

#         if loss_test < best_loss:
#             torch.save(model.state_dict(),f"./models_{tag}/model_step" + str(j))
#             best_ind = j
#             best_loss = loss_test
#         tic = time.time()
#     if j%200000 == 0:
#         model.eval()
#         run_model_visualization(model,x_test,y_test,s,device,figs_dir = tag)

# print(best_ind)

# # best_ind = 93000
# ## inference
# model.load_state_dict(torch.load(f'models_{tag}/model_step'+str(best_ind)))
# model.eval()
# run_model_visualization(model,x_test,y_test,s,device,figs_dir = tag)


