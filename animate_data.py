import scipy.io
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
data = scipy.io.loadmat('./training/ns_data.mat')
u = data["u"]
print(u.shape)


fig,axs = plt.subplots(1,5)
n = u.shape[-1]
ind = np.random.randint(0,u.shape[0],5)

vmin = np.min(u)
vmax = np.max(u)

print(vmin,vmax)

ims = []
for t in range(n):
    im = []
    for i in range(5):
        im.append(axs[i].imshow(u[ind[i],:,:,t],animated = 'True',cmap=plt.colormaps['turbo'],vmin=vmin, vmax=vmax))
    
    ims.append(im)
print(ims[0])
ani = animation.ArtistAnimation(fig,ims,interval = 1e-6)
ani.save("training/data_ani.gif")
    
