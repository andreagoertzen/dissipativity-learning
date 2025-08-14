import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

data = np.load('KS_data_l64.0.npz',allow_pickle=True)
physical_data = data['Physical']
physical_u = np.array(physical_data.item()['u'])
physical_x = np.array(physical_data.item()['x'])
physical_t = np.array(physical_data.item()['t'])

# u_data = torch.tensor(physical_u)
# U,S,V = torch.svd(u_data-torch.mean(u_data,0))
# mode1 = V[:,0]
# mode2 = V[:,1]
# x = torch.einsum('bi,i ->b',u_data,mode1).cpu()
# y = torch.einsum('bi,i ->b',u_data,mode2).cpu()
# plt.figure()
# plt.title('Data: System Trajectory in PCA Space (Attractor Projection)')
# sc = plt.scatter(x,y,c=physical_t,cmap='viridis',s=5)
# plt.colorbar(sc, label='Time T')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig('figs/pca_data')


u_model = torch.load('rolloutdata')
print(u_model.shape)
u_model = torch.tensor(u_model)
U,S,V = torch.svd(u_model-torch.mean(u_model,0))
mode1 = V[:,0]
mode2 = V[:,1]
x = torch.einsum('bi,i ->b',u_model,mode1).cpu()
y = torch.einsum('bi,i ->b',u_model,mode2).cpu()
plt.figure()
plt.title('Model: System Trajectory in PCA Space (Attractor Projection)')
sc = plt.scatter(x,y,c=np.linspace(0,u_model.shape[0],u_model.shape[0]),cmap='viridis',s=5)
plt.colorbar(sc, label='Time T')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('figs/pca_model')

