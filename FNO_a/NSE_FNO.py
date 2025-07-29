import numpy as np 
import scipy.io # type: ignore
import torch # type: ignore
from fno_2d import *
from timeit import default_timer
### try adding model.train() after
print('done')
data = scipy.io.loadmat('../training/ns_data_visc1e-3.mat')
u = data['u'] # trajectories
device = torch.device('cuda')
torch.manual_seed(10)
np.random.seed(10)
# device = torch.device('cpu')

modes = 20
width = 128

in_dim = 1
out_dim = 1

batch_size = 50
epochs = 50
learning_rate = 0.0005
scheduler_step = 10
scheduler_gamma = 0.5

S = u.shape[1]


def get_data(u):

    x = u[...,:-1].transpose(0,3,1,2)
    y = u[...,1:].transpose(0,3,1,2)

    x = x.reshape(-1,S,S)
    y = y.reshape(-1,S,S)

    x = torch.tensor(x).to(device)
    y = torch.tensor(y).to(device)

    return x,y


train_a,train_u = get_data(u[:150,:,:,100:])

print(train_a.shape)
print(train_u.shape)
test_a,test_u = get_data(u[150:,:,:,100:])

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

model = Net2d(in_dim, out_dim, S, modes, width).to(device)
# model.load_state_dict(torch.load('models/model_working'))
print(model.count_params())
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

loss_func = torch.nn.MSELoss().to(device)
def my_loss(outputs, targets):
    loss = torch.norm(outputs-targets)/torch.norm(targets)
    return loss

print('training...')
# Training

for ep in range(1, epochs + 1):
    model.train()
    t1 = default_timer()
    train_loss = 0
    for x, y in train_loader:
        # print(x.shape)
        # print(y.shape)
        x = x.to(device).reshape(x.shape[0], S, S, in_dim)
        y = y.to(device).reshape(y.shape[0], S, S, out_dim)

        out = model(x).reshape(y.shape[0], S, S, out_dim).to(device)
        loss = loss_func(out, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        # print(model.named_parameters())
        optimizer.step()

    test_metric = 0
    test_loss = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device).view(x.shape[0], S, S, in_dim)
            y = y.to(device).view(y.shape[0], S, S, out_dim)

            out = model(x).reshape(y.shape[0], S, S, out_dim)
            test_loss += loss_func(out,y)
            test_metric += my_loss(out, y).item()

    t2 = default_timer()
    scheduler.step()
    print("Epoch " + str(ep) + " completed in " + "{0:.{1}f}".format(t2-t1, 3) + " seconds. Train MSE:", "{0:.{1}f}".format(train_loss, 3), "Test MSE:", "{0:.{1}f}".format(test_loss, 3), "Test L2 err:",  "{0:.{1}f}".format(test_metric, 3))

torch.save(model, 'models/model_50epoch')