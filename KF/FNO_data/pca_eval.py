import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

M = N = 64

w_data_file = np.load('data/NS-Re40_T5000_dt0.001.npy')
w_data = w_data_file[:, 0, :, :].reshape(-1, 4096)
print(np.isnan(w_data).any())

w_data = torch.tensor(w_data).float()

U,S,V = torch.svd(w_data-torch.mean(w_data,0))

fig,axs = plt.subplots(2,5) 
for i in range(axs.shape[0]):
    for j in range(axs.shape[1]):
        axs[i][j].imshow(V[:,i*axs.shape[1]+j].reshape(M,N).cpu().numpy(), cmap=plt.colormaps['turbo'])

fig.suptitle('first 10 PCA modes of data',y=0.8)
fig.tight_layout()
plt.savefig(f'data/M64_Re40_PCA_data_5000.png')

plt.figure()
mode1 = V[:,0]
mode2 = V[:,1]

x = torch.einsum('bi,i ->b',w_data,mode1).cpu()
y = torch.einsum('bi,i ->b',w_data,mode2).cpu()

c = np.linspace(0,len(x),len(x))
print(c.shape)
print(x.shape)
plt.scatter(x.cpu().numpy(),y.cpu().numpy(),label='data',alpha=0.7,c=np.linspace(0,len(x),len(x)), cmap='viridis')
plt.colorbar()
# PLT.SHOW()
plt.savefig(f'data/M64_Re40_2PCA_modes_5000_b.png')

n_ani = 1
n_traj=1
t_save = 1
print('animating...')
w_save = w_data_file[:, 0, :, :]
ind = np.random.randint(0,n_traj,n_ani)
fig,axs = plt.subplots(1,n_ani)

if n_ani == 1:
    axs = [axs]
ims = []
with torch.no_grad():
    for t in range(w_save.shape[-1]):
        frame_artists = []
        for i in range(n_ani):
            im = axs[i].imshow(w_save[t, :, :],cmap='RdBu', animated=True)
            frame_artists.append(im)

        # Create a new text object every time
        time = t*t_save 
        title_obj = axs[i].text(0.5, 1.05, f"Time {time}", transform=axs[i].transAxes,
                                    ha='center', va='bottom', fontsize=12, animated=True)
        frame_artists.append(title_obj)

        ims.append(frame_artists)

ani = animation.ArtistAnimation(fig, ims, interval=1e-3)
ani.save(f"data/M64_Re40_data_ani_5000.gif")