import torch #type: ignore
import scipy.io #type: ignore
import scipy #type: ignore
import numpy as np 
import matplotlib.pyplot as plt #type: ignore
import seaborn as sns #type: ignore
import numpy.random as random
import utilities #type: ignore

device = torch.device('cuda')

data = scipy.io.loadmat('../training/ns_data_visc1e-3.mat')
u_data = data["u"]
s = u_data.shape[1]
print(type(u_data))
u_data = u_data[150:,:,:,100:].transpose(0,3,1,2).reshape(-1,s*s)
u_model = scipy.io.loadmat('inference/u_rollout.mat')['u']
print(type(u_model))


u_data = torch.from_numpy(u_data).to(device)
u_model = torch.from_numpy(u_model).to(device)

print(u_data.shape)
print(u_model.shape)

## PCA/POD
U,S,V = torch.svd(u_data-torch.mean(u_data,0))
U_model,S_model,V_model = torch.svd(u_model-torch.mean(u_model,0))

fig,axs = plt.subplots(2,5) 


for i in range(axs.shape[0]):
    for j in range(axs.shape[1]):
        axs[i][j].imshow(V[:,i*axs.shape[1]+j].reshape(s,s).cpu().numpy(), cmap=plt.colormaps['turbo'])

fig.suptitle('first 10 PCA modes of data',y=0.8)
fig.tight_layout()
plt.savefig('rollout_stats/PCA_data')

fig,axs = plt.subplots(2,5) 
for i in range(axs.shape[0]):
    for j in range(axs.shape[1]):
        axs[i][j].imshow(V_model[:,i*axs.shape[1]+j].reshape(s,s).cpu().numpy(), cmap=plt.colormaps['turbo'])
fig.suptitle('first 10 PCA modes of model',y=0.8)
fig.tight_layout()
plt.savefig('rollout_stats/PCA_model')

## Project data onto POD modes
plt.figure()
mode1 = V[:,0]
mode2 = V[:,1]

x = torch.einsum('bi,i ->b',u_data,mode1).cpu()
y = torch.einsum('bi,i ->b',u_data,mode2).cpu()

plt.plot(x,y,'.',label='data')

mode1 = V_model[:,0]
mode2 = V_model[:,1]

x_model = torch.einsum('bi,i ->b',u_model,mode1).cpu()
y_model = torch.einsum('bi,i ->b',u_model,mode2).cpu()
plt.plot(x_model,y_model,'r.',label ='model')
plt.savefig('rollout_stats/POD')

## POD Autocorrelation 

## PCA autocorrelation
plt.figure()
x1 = x[:2000]
x2 = x[:4000]
xc = np.correlate(np.array(x1),np.array(x2),mode='full')
plt.plot(xc[2000:4000],label='data')

x1 = x_model[:2000]
x2 = x_model[:4000]
xc = np.correlate(np.array(x1),np.array(x2),mode='full')
plt.plot(xc[2000:4000],label='model')
plt.legend()
plt.savefig('rollout_stats/PCA_autocorr')


## ENERGY SPECTRUM
u_new = u_data.reshape(-1,64,64)
# u_new = u_new[:60000,...]
print(u_new.shape)
u_fft = torch.fft.fft2(u_new).cpu()

k_max = u_new.shape[-1]//2
# print(k_max)

# torch fft returns X such that X[i] = conj(X[-i]), where i*2*pi is the frequency
wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1), \
                        torch.arange(start=-k_max, end=0, step=1)), 0).repeat(s, 1)

k_x = wavenumers.transpose(0, 1)
k_y = wavenumers

# Sum wavenumbers - for some reason they are saying k = k_x + k_y rather than sqrt(k_x**2 + k_y**2)
sum_k = torch.abs(k_x) + torch.abs(k_y)
sum_k = sum_k.numpy()
prod_k = torch.sqrt(k_x**2+k_y**2).int().numpy()


# Remove symmetric components from wavenumbers 
index = -1.0 * np.ones((s, s))
# print(index)
index[0:k_max + 1, 0:k_max + 1] = sum_k[0:k_max + 1, 0:k_max + 1]
# print(index)
# index = sum_k 

spectrum = np.zeros((u_fft.shape[0], s))
for j in range(1, s + 1):
    ind = np.where(index == j) # change index to prod_k to match bottom
    # print(ind)
    spectrum[:, j - 1] = np.sqrt( (u_fft[:, ind[0], ind[1]].sum(axis=1)).abs() ** 2) # change this to the one below to match bottom
    # spectrum[:, j - 1] =  ((u_fft[:, ind[0], ind[1]]).abs()**2).sum(axis=1) 
spectrum = spectrum.mean(axis=0)
# print(spectrum.shape)

u_new = u_model.reshape(-1,64,64)
u_new = u_model.reshape(-1,64,64)
print(u_new.shape)
u_fft = torch.fft.fft2(u_new).cpu()

spectrum_model = np.zeros((u_fft.shape[0], s))
for j in range(1, s + 1):
    ind = np.where(index == j) # change index to prod_k to match bottom
    # print(ind)
    spectrum_model[:, j - 1] = np.sqrt( (u_fft[:, ind[0], ind[1]].sum(axis=1)).abs() ** 2) # change this to the one below to match bottom
    # spectrum[:, j - 1] =  ((u_fft[:, ind[0], ind[1]]).abs()**2).sum(axis=1) 
spectrum_model = spectrum_model.mean(axis=0)
# print(spectrum.shape)



fig, ax = plt.subplots()
ax.plot(spectrum[:s//2],label = 'data')
ax.plot(spectrum_model[:s//2],label = 'model')
ax.set_yscale('log')
plt.xlabel('wave number')
plt.ylabel('energy')
plt.legend()
plt.savefig('rollout_stats/energyspectrum')


# velocity/vorticity dist
u_data = u_data.cpu()
u_model = u_model.cpu()
fig = plt.figure()    
ax = fig.add_subplot(121)
sns.kdeplot(u_data.reshape(-1)[random.permutation(len(u_data.reshape(-1)))[:50000]],ax = ax,label = 'data')
sns.kdeplot(u_model.reshape(-1)[random.permutation(len(u_model.reshape(-1)))[:50000]],ax = ax, label = 'model')
plt.legend()
plt.title('Pixelwise vorticity distribution')

v_data = utilities.w_to_u(u_data.reshape(-1,s,s)[:10000])
v_model = utilities.w_to_u(u_model.reshape(-1,s,s))

ax = fig.add_subplot(122)
sns.kdeplot(v_data.reshape(-1)[random.permutation(len(v_data.reshape(-1)))[:5000]],ax = ax,label = 'data')
sns.kdeplot(v_model.reshape(-1)[random.permutation(len(v_model.reshape(-1)))[:50000]],ax = ax,label = 'model')
plt.title('Pixelwise velocity distribution')
plt.legend()
plt.tight_layout()
plt.savefig('rollout_stats/distributions')

#3 spatial correlation
print(u_data.shape)
u_corr = u_data.reshape(-1,64,64)

fig,axs = plt.subplots(1,2) 
corr2 = np.zeros(u_corr.shape)
for i in range(u_corr.shape[-1]):
    corr2[i,...] = scipy.signal.correlate2d(u_corr[i,...],u_corr[i,...],mode = 'same',boundary = 'wrap')
# print(corr2.shape)
corr2 = np.mean(corr2,axis=0)
# print(corr2.shape)
im = axs[0].imshow(corr2)
fig.colorbar(im,shrink=0.6)
axs[0].set_title('data')
# plt.show()

u_corr = u_model.reshape(-1,64,64)
corr2 = np.zeros(u_corr.shape)
for i in range(u_corr.shape[-1]):
    corr2[i,...] = scipy.signal.correlate2d(u_corr[i,...],u_corr[i,...],mode = 'same',boundary = 'wrap')
# print(corr2.shape)
corr2 = np.mean(corr2,axis=0)
# print(corr2.shape)
im = axs[1].imshow(corr2)
fig.colorbar(im,shrink=0.6)
axs[1].set_title('model')
plt.suptitle('spatial correlation',y=0.8)
plt.tight_layout()
plt.savefig('rollout_stats/spatialcorr')


