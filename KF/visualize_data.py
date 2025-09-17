import torch
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
folder = 'KF_Re500_M128_tsave0.5_T500_n200'
w_save = torch.load(f'data/{folder}/data.pt')
M = N = 128

## PCA for entire traj
print('PCA...')
w_data = w_save[:,:,:,10:].permute(0,3,1,2).reshape(-1,M*N)

print(w_data.shape)

U,S,V = torch.svd(w_data-torch.mean(w_data,0))

fig,axs = plt.subplots(2,5) 
for i in range(axs.shape[0]):
    for j in range(axs.shape[1]):
        axs[i][j].imshow(V[:,i*axs.shape[1]+j].reshape(M,N).cpu().numpy(), cmap=plt.colormaps['turbo'])

fig.suptitle('first 10 PCA modes of data',y=0.8)
fig.tight_layout()
plt.savefig(f'data/{folder}/PCA_data.png')

plt.figure()
mode1 = V[:,0]
mode2 = V[:,1]

x = torch.einsum('bi,i ->b',w_data,mode1).cpu()
y = torch.einsum('bi,i ->b',w_data,mode2).cpu()

plt.plot(x,y,'.',label='data',alpha=0.7)
plt.savefig(f'data/{folder}/2PCA_modes.png')