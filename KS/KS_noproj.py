import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import jax

device = torch.device('cuda')
torch.manual_seed(0)

data = np.load('KS_data_l64.0.npz',allow_pickle=True)
physical_data = data['Physical']
physical_u = np.array(physical_data.item()['u'])
physical_x = np.array(physical_data.item()['x'])
physical_t = np.array(physical_data.item()['t'])

print(physical_u.shape)
print(physical_x.shape)
print(physical_t.shape)

dt = 0.1

def get_data(u,x):
    u = u[0::10,...]
    x_data = (torch.tensor(u[:-1,...]).to(device),torch.tensor(x.reshape(len(x),1)).to(device))
    y_data = torch.tensor(u[1:,...]).to(device)

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
        self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(128 * 4 * 4, 128) 
        # self.fc1 = nn.Linear(1280, 256) 
        # self.fc1 = nn.Linear(1664, 256) 
        # self.fc1 = nn.Linear(1856, 256) 
        # self.fc1 = nn.Linear(1984, 256) 
        self.fc1 = nn.Linear(128,128)
        self.fc2 = nn.Linear(128, 128)

    def forward(self, x):
        x = self.reshape(x)
        # x = self.activation(self.conv1(x))
        # x = self.activation(self.conv2(x))
        # x = self.activation(self.conv3(x))
        # x = self.activation(self.conv4(x))
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

class Trunk(nn.Module):
    def __init__(self, m, activation=F.relu):
        super(Trunk, self).__init__()
        self.m = m
        self.activation = activation

        self.fc1 = nn.Linear(m, 128) 
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        # x = self.fc2(x)
        # x = self.activation(x)
        # x = self.fc3(x)
        # x = self.activation(x)
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

def get_batch(x,y,bsize):
    ind_time = np.random.randint(0,x[0].shape[0],bsize)
    x_out1 = x[0][ind_time,:]
    x_batch = (x_out1.to(device),x[1].to(device))
    y_batch = y[ind_time,:].to(device)
    return x_batch,y_batch


x_test,y_test = get_data(physical_u[90000:,...],physical_x)
x_train,y_train = get_data(physical_u[:90000,...],physical_x)
print(y_test.shape)
print(y_train.shape)


s = physical_u.shape[1]
branch = Branch(m=s**2).to(device)
trunk = Trunk(m=1).to(device)
model = DeepONet(branch,trunk).to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)#,weight_decay=1e-4)
num_params = sum(v.numel() for v in model.parameters() if v.requires_grad)
print(num_params)

loss_func = torch.nn.MSELoss()
tic = time.time()

iterations = 1000000
n_save = 1000

bsize = 10000
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
        plt.savefig(f"figs/loss_iter.png")
        plt.close()

        if loss_test < best_loss:
            torch.save(model.state_dict(),"./models/model_step" + str(j))
            best_ind = j
            best_loss = loss_test
        tic = time.time()
    if j%200000 == 0:
        model.eval()

        # 1 time step rollout (on test data)
        u_test = model(x_test)
        fig,axs = plt.subplots(2,1)
        print(y_test.shape)
        print(u_test.shape)
        axs[0].imshow(y_test.T.detach().cpu().numpy().astype(np.float32),extent=[physical_t[90000]*dt,physical_t[-1]*dt,physical_x[0],physical_x[-1]],aspect='auto',vmin=-2.5,vmax=2.5)
        axs[0].set_title('data')
        axs[0].set_xlabel('time (s)')
        axs[0].set_ylabel('position')
        im = axs[1].imshow(u_test.T.detach().cpu().numpy().astype(np.float32),extent=[physical_t[90000]*dt,physical_t[-1]*dt,physical_x[0],physical_x[-1]],aspect='auto',vmin=-2.5,vmax=2.5)
        axs[1].set_title('model')
        axs[1].set_xlabel('time (s)')
        axs[1].set_ylabel('position')
        fig.tight_layout()
        fig.colorbar(im,ax=axs,location='right')
        plt.savefig('figs/1step')

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
        plt.savefig('figs/rollout')

        # rollout on random IC
        n_rollout = 10000
        key = jax.random.PRNGKey(10)*5
        u0 = jax.random.normal(key, (1,s)) 
        print(u0.shape)
        rollout_traj = torch.zeros(n_rollout,s)
        u_out = torch.tensor(np.array(u0)).to(device)
        for i in range(n_rollout):
            u_out = model((u_out,x_test[1]))
            rollout_traj[i,:] = u_out
        plt.figure()
        plt.imshow(rollout_traj.T.detach().numpy().astype(np.float32), extent=[0,n_rollout,0,s],aspect='auto')
        plt.colorbar()
        plt.xlabel('time')
        plt.ylabel('position')
        plt.savefig('figs/rollout_randomIC')
        model.train()

print(best_ind)

# best_ind = 93000
## inference
model.load_state_dict(torch.load('models/model_step'+str(best_ind)))
model.eval()

# 1 time step rollout (on test data)
u_test = model(x_test)
fig,axs = plt.subplots(2,1)
print(y_test.shape)
print(u_test.shape)
axs[0].imshow(y_test.T.detach().cpu().numpy().astype(np.float32),extent=[physical_t[90000]*dt,physical_t[-1]*dt,physical_x[0],physical_x[-1]],aspect='auto',vmin=-2.5,vmax=2.5)
axs[0].set_title('data')
axs[0].set_xlabel('time (s)')
axs[0].set_ylabel('position')
im = axs[1].imshow(u_test.T.detach().cpu().numpy().astype(np.float32),extent=[physical_t[90000]*dt,physical_t[-1]*dt,physical_x[0],physical_x[-1]],aspect='auto',vmin=-2.5,vmax=2.5)
axs[1].set_title('model')
axs[1].set_xlabel('time (s)')
axs[1].set_ylabel('position')
fig.tight_layout()
fig.colorbar(im,ax=axs,location='right')
plt.savefig('figs/1step')

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
plt.savefig('figs/rollout')

# rollout on random IC
n_rollout = 10000
key = jax.random.PRNGKey(10)*5
u0 = jax.random.normal(key, (1,s)) 
print(u0.shape)
rollout_traj = torch.zeros(n_rollout,s)
u_out = torch.tensor(np.array(u0)).to(device)
for i in range(n_rollout):
    u_out = model((u_out,x_test[1]))
    rollout_traj[i,:] = u_out
plt.figure()
plt.imshow(rollout_traj.T.detach().numpy().astype(np.float32), extent=[0,n_rollout,0,s],aspect='auto')
plt.colorbar()
plt.xlabel('time')
plt.ylabel('position')
plt.savefig('figs/rollout_randomIC')
# torch.save(rollout_traj,'rolloutdata')




