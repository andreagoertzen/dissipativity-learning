import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import jax
import h5py

device = torch.device('cuda')

f = h5py.File('KS.mat')
arrays = {}
for k,v in f.items():
    arrays[k] = np.array(v)
u_MNO = arrays['u']
dt = 1

def get_data(u,x):
    if len(u.shape) == 3:
        u = u.transpose(1,2,0)
    elif len(u.shape) == 2:
        u = u.transpose(1,0)
    u_x = u[:-1,...].reshape(-1,u.shape[-1])
    u_x = torch.tensor(u_x,dtype=torch.float32).to(device)
    u_y = torch.tensor(u[1:,...].reshape(-1,u.shape[-1]),dtype=torch.float32).to(device)
    x = torch.tensor(x.reshape(len(x),1),dtype=torch.float32).to(device)

    print(type(u_x),type(u_y),type(x))
    x_data = (u_x,x)
    y_data = u_y
    return x_data,y_data


class Branch(nn.Module):
    def __init__(self, m, activation=F.relu):
        super(Branch, self).__init__()
        self.m = m
        self.activation = activation

        # self.reshape = lambda x: x.view(-1, 1, 28, 28)
        self.reshape = lambda x: x.view(-1, 1, s)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=2)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=2)
        self.conv5 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5, stride=2)
        self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(128 * 4 * 4, 128) 
        # self.fc1 = nn.Linear(1280, 256) 
        # self.fc1 = nn.Linear(1856, 256) 
        self.fc1 = nn.Linear(7424, 512) 
        self.fc2 = nn.Linear(512, 1024)

    def forward(self, x):
        x = self.reshape(x)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        # x = self.activation(self.conv5(x))
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

class Trunk(nn.Module):
    def __init__(self, m, activation=F.relu):
        super(Trunk, self).__init__()
        self.m = m
        self.activation = activation

        self.fc1 = nn.Linear(m, 1024) 
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 1024)
        self.fc6 = nn.Linear(1024, 1024)
        self.fc7 = nn.Linear(1024, 1024)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)
        x = self.activation(x)
        x = self.fc5(x)
        x = self.activation(x)
        x = self.fc6(x)
        x = self.activation(x)
        x = self.fc7(x)
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

def get_batch(x,y,bsize):
    ind_time = np.random.randint(0,x[0].shape[0],bsize)
    # ind_pos = np.random.randint(0,x[1].shape[0],int(x[1].shape[0]/2))
    x_out1 = x[0][ind_time,:]
    x_batch = (x_out1.to(device),x[1].to(device))
    y_batch = y[ind_time,:].to(device)
    # y_batch = y_batch[:,ind_pos]
    return x_batch,y_batch

physical_x = torch.linspace(0,32*np.pi,u_MNO.shape[0])
physical_x = torch.linspace(0,u_MNO.shape[0],u_MNO.shape[0])*0.01
# x_test,y_test = get_data(u_MNO[:,300:,1001],physical_x)
x_test,y_test = get_data(u_MNO[:,:,1001],physical_x)
x_train,y_train = get_data(u_MNO[:,:,:1000],physical_x)
print(x_train[0].shape)
print(y_test.shape)
print(y_train.shape)


s = u_MNO.shape[0]
branch = Branch(m=s**2).to(device)
trunk = Trunk(m=1).to(device)
model = DeepONet(branch,trunk).to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)#,weight_decay=1e-4)
num_params = sum(v.numel() for v in model.parameters() if v.requires_grad)
print(num_params)

loss_func = torch.nn.MSELoss()
tic = time.time()

iterations = 100000
n_save = 1000

bsize = 1000
best_loss = 1e5
model.train()

losses = []
test_losses = []

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
        # loss_metric = my_loss(u_test,y_test).to("cpu").detach().numpy()
        print(f"iteration: {j}      train loss: {loss:.2e}      test loss: {loss_test:.2e}")#      test metric: {loss_metric:.2e}")
        print(total_time)
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
        plt.savefig(f"figs/loss_iter_MNO2.png")
        plt.close()

        if loss_test < best_loss:
            torch.save(model.state_dict(),"./models_MNO/model_step" + str(j))
            best_ind = j
            best_loss = loss_test
        tic = time.time()

print(best_ind)

# best_ind = 93000

## inference
model.load_state_dict(torch.load('models_MNO/model_step'+str(best_ind)))
model.eval()

# 1 time step rollout (on test data)
u_test = model(x_test)
fig,axs = plt.subplots(2,1)
print(y_test.shape)
print(u_test.shape)
axs[0].imshow(y_test.T.detach().cpu().numpy().astype(np.float32),aspect='auto')
axs[0].set_title('data')
axs[1].imshow(u_test.T.detach().cpu().numpy().astype(np.float32),aspect='auto')
axs[1].set_title('model')
plt.tight_layout()
plt.savefig('figs/1step_MNO2')

# long trajectory rollout (on test data)
n_rollout = 1000
rollout_traj = torch.zeros(n_rollout,s)
u_out = x_test[0][0,...]
for i in range(n_rollout):
    u_out = model((u_out,x_test[1]))
    rollout_traj[i,:] = u_out
plt.figure()
plt.imshow(rollout_traj.T.detach().numpy().astype(np.float32), extent=[0,n_rollout,0,s],aspect='auto')
plt.colorbar()
plt.savefig('figs/rollout_MNO2')

# rollout on random IC
n_rollout = 1000
key = jax.random.PRNGKey(10)
u0 = jax.random.normal(key, (1,s)) 
print(u0.shape)
rollout_traj = torch.zeros(n_rollout,s)
u_out = torch.tensor(np.array(u0)).to(device)
for i in range(n_rollout):
    u_out = model((u_out,x_test[1]))
    rollout_traj[i,:] = u_out
plt.figure()
plt.imshow(rollout_traj.T.detach().numpy().astype(np.float32), extent=[0,n_rollout,0,s],aspect='auto')
plt.xlabel('time')
plt.ylabel('position')
plt.colorbar()
plt.savefig('figs/rollout_randomIC_MNO2')
# torch.save(rollout_traj,'rolloutdata')