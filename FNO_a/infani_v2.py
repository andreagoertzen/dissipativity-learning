import scipy # type: ignore
import torch # type: ignore
import itertools

import numpy as np
import matplotlib.pyplot as plt # type: ignore
import matplotlib.animation as animation # type: ignore

## error possibly related to loading model rather than model state_dict
S = 64
device = torch.device("cuda")

def get_data(u):

    x = u[...,:-1].transpose(0,3,1,2)
    y = u[...,1:].transpose(0,3,1,2)

    x = x.reshape(-1,S,S)
    y = y.reshape(-1,S,S)

    x = torch.tensor(x).to(device)
    y = torch.tensor(y).to(device)

    return x,y

data = scipy.io.loadmat('../training/ns_data_visc1e-3.mat')
u = data['u'] # trajectories

model = torch.load('models/model_50epoch',map_location=device)
model.eval()

data = u[150:,:,:,100:]
x_val,y_val = get_data(data)

## add comparison vs test (1 time step)
val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_val, y_val), batch_size=50, shuffle=False)
pred = np.empty([1,S,S])
print(pred.shape)
val = np.empty([1,S,S])
print(val.shape)
with torch.no_grad():
    for x, y in val_loader:
        pred = np.concatenate((pred,model(x.reshape(-1,S,S,1)).reshape(-1,S,S).cpu().numpy()),axis=0)
        val = np.concatenate((val,y.cpu().numpy()),axis=0)
scipy.io.savemat('inference/u_1step.mat', mdict={'u_data': val,'u_model':pred})


## animation to compare to a single trajectory
data = u[150:,:,:,100:]
x_val,y_val = get_data(data)
ims = []
n_times = u.shape[-1]-101
fig,axs = plt.subplots(2)  
axs[0].set_title('Data') 
axs[1].set_title('Model')
fig.tight_layout()  

vmin = torch.min(y_val)
vmax = torch.max(y_val)
print(vmin,vmax)

## animate
with torch.no_grad():
    y_pred = model(x_val[0,...].reshape(1,S,S,1))
print(y_pred.shape)
with torch.no_grad():
    for i in range(n_times):
        # print(i)

        im = axs[0].imshow(y_val[i,:].reshape(S,S).cpu().numpy(),animated = 'True',cmap=plt.colormaps['turbo'],vmin=vmin, vmax=vmax)
        im2 = axs[1].imshow(y_pred.reshape(S,S).cpu().numpy(),animated = 'True',cmap=plt.colormaps['turbo'],vmin=vmin,vmax=vmax)
        y_pred = model(y_pred)

        ims.append([im,im2])

ani = animation.ArtistAnimation(fig,ims,interval = 1e-6)
ani.save("inference/compare.gif")#, writer = 'ffmpeg')


## predict long rollout
fig = plt.figure()    
ax = fig.add_subplot(111)
ims = []
n_compose = 10000
u_model = np.zeros([n_compose,S,S])
y_pred = model(x_val[0,...].reshape(1,S,S,1))
print(y_pred.shape)
with torch.no_grad():
    for i in range(n_compose):
        im = plt.imshow(y_pred.reshape(S,S).cpu().numpy(),animated = 'True',cmap=plt.colormaps['turbo'],vmin=vmin, vmax=vmax)
        ims.append([im])
        u_model[i,...] = y_pred.reshape(S,S).cpu().numpy()
        y_pred = model(y_pred)

scipy.io.savemat('inference/u_rollout.mat', mdict={'u': u_model})
ani = animation.ArtistAnimation(fig,ims,interval = 1e-6)
ani.save("inference/longmodel.gif")#, writer = 'ffmpeg')
