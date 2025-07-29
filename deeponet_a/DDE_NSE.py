import numpy as np #type:ignore
import scipy #type:ignore
import torch #type:ignore
import time
import deepxde as dde #type:ignore
import torch.nn as nn #type:ignore
import torch.nn.functional as F #type:ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.animation as animation # type: ignore

## uses deepxde package (would be hard to add projection layer)

data = scipy.io.loadmat('../training/ns_data_visc1e-3.mat')
u = data['u'] # trajectories
device = torch.device('cuda')

nt = 1
tout = 1
def get_data(u):
    
    s = u.shape[1]

    u_branch = u[...,:-1].transpose(0,3,1,2).reshape(-1,s*s)
    y = u[...,1:].transpose(0,3,1,2).reshape(-1,s*s)
    grids = []
    grids.append(np.linspace(0, 1, s, dtype=np.float32))
    grids.append(np.linspace(1, 0, s, dtype=np.float32)) # position (0,0) of matrix is point (0,1) on plot (top left)

    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T 

    x_train = (u_branch.astype(np.float32), grid.astype(np.float32))
    y_train = y.astype(np.float32)

    return x_train,y_train

x_train,y_train = get_data(u[:150,:,:,100:])
x_test,y_test = get_data(u[150:,:,:,100:])

data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)
m = 28**2
m = 64**2*nt
activation = "relu"

class Branch(nn.Module):
    def __init__(self, m, activation=F.relu):
        super(Branch, self).__init__()
        self.m = m
        self.activation = activation

        # self.reshape = lambda x: x.view(-1, 1, 28, 28)  
        self.reshape = lambda x: x.view(-1, nt, 64, 64)
        self.conv1 = nn.Conv2d(in_channels=nt, out_channels=64, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2)
        self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(128 * 4 * 4, 128) 
        self.fc1 = nn.Linear(512 * 1 * 1, 256) 
        self.fc2 = nn.Linear(256, 256)

    def forward(self, x):
        x = self.reshape(x)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


net = dde.maps.DeepONetCartesianProd(
    [m, Branch(m)], [2, 256, 256, 256, 256], activation, "Glorot normal"
)
model = dde.Model(data, net)
model.compile(
        "adam",lr=3e-4,
        # decay=("inverse time", 1, 1e-4),
        metrics=["mean l2 relative error"],
    )


checker = dde.callbacks.ModelCheckpoint(
    "models/model.ckpt", save_better_only=True, period=1000
)


losshistory, train_state = model.train(epochs=100000, batch_size=990,callbacks=[checker])
print(train_state.best_step)


model.restore("models/model.ckpt-" + str(train_state.best_step) + ".pt", verbose=1)


s = 64

#### INFERENCE
x_val,y_val = get_data(u[150:,:,:,100:])

## add comparison vs test (1 time step)
y_pred = model.predict((x_val[0],x_val[1]))
print(y_pred.shape)
print(y_val.shape)
scipy.io.savemat('inference/u_1step.mat', mdict={'u_data': y_val,'u_model':y_pred})


ims = []
n_times = u.shape[-1]-101
fig,axs = plt.subplots(2)  
axs[0].set_title('Data') 
axs[1].set_title('Model')
fig.tight_layout()  

vmin = np.min(y_val)
vmax = np.max(y_val)

## animation to compare to a single trajectory
y_pred = model.predict((x_val[0][0,...],x_val[1]))
for i in range(n_times):

    # im = axs[0].imshow(u[154,:,:,i+1],animated = 'True',cmap=plt.colormaps['turbo'])
    im = axs[0].imshow(y_val[i,:].reshape(s,s),animated = 'True',cmap=plt.colormaps['turbo'],vmin=vmin, vmax=vmax)#vmin=np.min(y_val[i,:]),vmax=np.max(y_val[i,:]))
    im2 = axs[1].imshow(y_pred.reshape(s,s),animated = 'True',cmap=plt.colormaps['turbo'],vmin=vmin,vmax=vmax)#vmin=np.min(y_val[i,:]),vmax=np.max(y_val[i,:]))
    y_pred = model.predict((y_pred,x_val[1]))

    ims.append([im,im2])

ani = animation.ArtistAnimation(fig,ims,interval = 1e-6)
ani.save("inference/compare.gif")#, writer = 'ffmpeg')


## predict long rollout
fig = plt.figure()    
ax = fig.add_subplot(111)
ims = []
n_compose = 10000
u_model = np.zeros([n_compose,s*s])
y_pred = model.predict((x_val[0][0,...],x_val[1]))
for i in range(n_compose):
    im = plt.imshow(y_pred.reshape(s,s),animated = 'True',cmap=plt.colormaps['turbo'],vmin=vmin, vmax=vmax)
    ims.append([im])
    u_model[i,...] = y_pred
    y_pred = model.predict((y_pred,x_val[1]))

scipy.io.savemat('inference/u_rollout.mat', mdict={'u': u_model})
ani = animation.ArtistAnimation(fig,ims,interval = 1e-6)
ani.save("inference/longmodel.gif")#, writer = 'ffmpeg')
