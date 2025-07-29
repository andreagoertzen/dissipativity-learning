import numpy as np
import scipy
import torch
import time
import deepxde as dde
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# uses pytorch rather than deepxde package so we can add projection layer

device = torch.device('cuda')
torch.manual_seed(0)
np.random.seed(0)

data = scipy.io.loadmat('../training/ns_data_visc1e-3.mat')
u = data['u'] 
t = data['t']


def get_data(u):
    
    s = u.shape[1]
    u_branch = u[...,:-1].transpose(0,3,1,2).reshape(-1,s*s)
    y = u[...,1:].transpose(0,3,1,2).reshape(-1,s*s)
    grids = []
    grids.append(np.linspace(0, 1, s, dtype=np.float32))
    grids.append(np.linspace(1, 0, s, dtype=np.float32)) # position (0,0) of matrix is point (0,1) on plot (top left)

    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T 

    x_train = (torch.tensor(u_branch).to(device), torch.tensor(grid).to(device))
    y_train = torch.tensor(y).to(device)

    return x_train,y_train

print('getting data...')
x_test,y_test = get_data(u[150:,:,:,100:])
x_train,y_train = get_data(u[:150,:,:,100:])
print('done')


# data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)
# m = 28**2
m = 64**2
activation = "relu"
nt = 1
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

class Trunk(nn.Module):
    def __init__(self, m, activation=F.relu):
        super(Trunk, self).__init__()
        self.m = m
        self.activation = activation

        self.fc1 = nn.Linear(m, 256) 
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)
        x = self.activation(x)
        return x

class DeepONet(nn.Module):
    def __init__(self,Branch,Trunk):
        super(DeepONet,self).__init__()

        self.Branch = Branch
        self.Trunk = Trunk

        self.b = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self,x):
        x1 = self.Branch(x[0])
        x2 = self.Trunk(x[1])
        x_out = torch.einsum("ai,bi->ab",x1,x2)
        x_out += self.b
        return x_out

print('initializing model...')


branch = Branch(m=u.shape[1]**2).to(device)
trunk = Trunk(m=2).to(device)
model = DeepONet(branch,trunk).to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)#,weight_decay=1e-4)
num_params = sum(v.numel() for v in model.parameters() if v.requires_grad)
print(num_params)

def get_batch(x,y,bsize):
    ind_time = np.random.randint(0,x[0].shape[0],bsize)
    x_out1 = x[0][ind_time,:]
    x_batch = (x_out1.to(device),x[1].to(device))
    y_batch = y[ind_time,:].to(device)
    return x_batch,y_batch

def my_loss(outputs, targets):
    loss = torch.mean(torch.norm(outputs-targets)/torch.norm(targets))
    return loss

loss_func = torch.nn.MSELoss()
tic = time.time()

iterations = 100000
n_save = 1000

bsize = 990
best_loss = 1e5
model.train()
for j in range(iterations+1):

    x_batch,y_batch = get_batch(x_train,y_train,bsize=bsize)
    optimizer.zero_grad() 
    u_pred2 = model(x_batch)#.unsqueeze(1)
    loss = loss_func(u_pred2,y_batch)
    loss.backward()
    optimizer.step()

    if j%n_save == 0:
        toc = time.time()
        total_time = (toc-tic)
        u_test = model(x_test)#.unsqueeze(1)
        loss = loss.to("cpu").detach().numpy()
        loss_test = loss_func(u_test,y_test)
        loss_test = loss_test.to("cpu").detach().numpy()
        loss_metric = my_loss(u_test,y_test).to("cpu").detach().numpy()
        print(f"iteration: {j}      train loss: {loss:.2e}      test loss: {loss_test:.2e}      test metric: {loss_metric:.2e}")
        print(total_time)

        if loss_test < best_loss:
            torch.save(model.state_dict(),"./models/model_step" + str(j))
            best_ind = j
            best_loss = loss_test
        tic = time.time()

print(best_ind)

model.load_state_dict(torch.load('models/model_step'+str(best_ind)))

model.eval()
s = 64


#### INFERENCE
x_val,y_val = get_data(u[150:,:,:,100:])

## add comparison vs test (1 time step)
with torch.no_grad():
    y_pred = model((x_val[0],x_val[1]))
print(y_pred.shape)
print(y_val.shape)
scipy.io.savemat('inference/u_1step.mat', mdict={'u_data': y_val.cpu(),'u_model':y_pred.cpu()})


ims = []
n_times = u.shape[-1]-101
fig,axs = plt.subplots(2)  
axs[0].set_title('Data') 
axs[1].set_title('Model')
fig.tight_layout()  

vmin = torch.min(y_val)
vmax = torch.max(y_val)

## animation to compare to a single trajectory
with torch.no_grad():
    y_pred = model((x_val[0][0,...],x_val[1]))
    for i in range(n_times):

        # im = axs[0].imshow(u[154,:,:,i+1],animated = 'True',cmap=plt.colormaps['turbo'])
        im = axs[0].imshow(y_val[i,:].reshape(s,s).cpu(),animated = 'True',cmap=plt.colormaps['turbo'],vmin=vmin, vmax=vmax)#vmin=np.min(y_val[i,:]),vmax=np.max(y_val[i,:]))
        im2 = axs[1].imshow(y_pred.reshape(s,s).cpu(),animated = 'True',cmap=plt.colormaps['turbo'],vmin=vmin,vmax=vmax)#vmin=np.min(y_val[i,:]),vmax=np.max(y_val[i,:]))
        y_pred = model((y_pred,x_val[1]))

        ims.append([im,im2])

ani = animation.ArtistAnimation(fig,ims,interval = 1e-6)
ani.save("inference/compare.gif")#, writer = 'ffmpeg')


## predict long rollout
fig = plt.figure()    
ax = fig.add_subplot(111)
ims = []
n_compose = 10000
u_model = torch.zeros([n_compose,s*s])

with torch.no_grad():
    y_pred = model((x_val[0][0,...],x_val[1]))
    for i in range(n_compose):
        im = plt.imshow(y_pred.reshape(s,s).cpu(),animated = 'True',cmap=plt.colormaps['turbo'],vmin=vmin, vmax=vmax)
        ims.append([im])
        u_model[i,...] = y_pred
        y_pred = model((y_pred,x_val[1]))

scipy.io.savemat('inference/u_rollout.mat', mdict={'u': u_model.cpu()})
ani = animation.ArtistAnimation(fig,ims,interval = 1e-6)
ani.save("inference/longmodel.gif")#, writer = 'ffmpeg')
