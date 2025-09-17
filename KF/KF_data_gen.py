import torch
import numpy as np
from gstools import SRF, Gaussian
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from tqdm import tqdm

##################################################
## Training data generation for Kolmogorov flow ##
#################################################
device = torch.device('cuda')
Re = 250
Re = 500 # Reynolds number
# Re = 70
dt = 0.0005 # Integration time step
n = 4 # forcing period 
T = 500 # end time
M = N = 128 # x and y discretization
t_save = 0.5 # save time step
n_traj = 50 # number of trajectories to generate 
n_ani = 5 # how many trajectories to visualize
ic_factor = 1

domain_size = L = 2 * np.pi
dx = domain_size/M
dy = domain_size/N
kx = torch.fft.fftfreq(N, dx).to(device)
ky = torch.fft.fftfreq(M, dy).to(device)
kx, ky = torch.meshgrid(kx,ky, indexing='ij')
n_steps = int(T/dt)
x = torch.linspace(0, domain_size, N, dtype=torch.float32).to(device)
y = torch.linspace(0, domain_size, N, dtype=torch.float32).to(device)
X, Y = torch.meshgrid(x, y, indexing='ij')

dealias = torch.ones(N,N).to(device)
dealias[kx*L>N/3-1] = 0 
dealias[kx*L<-N/3] = 0 
dealias[ky*L>N/3-1] = 0 
dealias[ky*L<-N/3] = 0 

folder = f'KF_Re{Re}_M{M}_tsave{t_save}_T{T}_n{n_traj}'
if not os.path.exists(f'data/{folder}'):
    os.makedirs(f'data/{folder}')


def linear_terms(omega_hat):
    return 1/Re * (2j *np.pi)**2 * (kx**2 + ky**2) * omega_hat
    

def nonlinear_terms(omega_hat,forcing=1):
    double_derivative = (2*np.pi*1j)**2 * ( kx**2 + ky**2 )
    double_derivative[0,0] = 1.0

    psi_hat = -1 * omega_hat/double_derivative
    uhat = 2j*np.pi*ky*psi_hat
    vhat = -2j*np.pi*kx*psi_hat

    u,v = torch.fft.ifft2(uhat).real, torch.fft.ifft2(vhat).real

    # grad_x_hat = 2j*np.pi*kx*omega_hat
    # grad_y_hat = 2j*np.pi*ky*omega_hat
    # grad_x, grad_y = torch.fft.ifft2(grad_x_hat).real, torch.fft.ifft2(grad_y_hat).real

    # advection = -(grad_x*u + grad_y*v)
    # advection_hat = torch.fft.fft2(advection)

    omega = torch.fft.ifft2(omega_hat).real
    advection_hat = -2j *np.pi* (kx*torch.fft.fft2(u*omega) + ky*torch.fft.fft2(v*omega))
    advection_hat = advection_hat*dealias

    forcing_x = torch.sin(n * Y)
    forcing_y = (0*Y)
    fx_hat, fy_hat = torch.fft.fft2(forcing_x), torch.fft.fft2(forcing_y)
    derivative_term = (2j*np.pi)
    if forcing==1:
        forcing_hat = derivative_term * (fy_hat*kx - fx_hat*ky)
    elif forcing==2:
        forcing_hat = -torch.fft.fft2(n*torch.cos(n*Y))
    # forcing_hat = torch.fft.fft2(0.1*(torch.sin(X + Y) + torch.cos(X + Y)))

    return advection_hat + forcing_hat

def euler_step(omega_hat,dt):
    return omega_hat + (linear_terms(omega_hat)+nonlinear_terms(omega_hat,forcing=1))*dt


def euler_step2(omega_hat,dt):
    return omega_hat + (linear_terms(omega_hat)+nonlinear_terms(omega_hat,forcing=2))*dt

def update_step(omega_hat,nonlinear_terms,dt):
    # from https://arxiv.org/pdf/1207.4682
    ## eq 3.4
    # G = 1/Re * (2j *np.pi)**2 * (kx**2 + ky**2)
    G = -1/Re * (2j *np.pi)**2 * (kx**2 + ky**2)
    # (omega_tilde-omega_hat)/dt = -G/2(omega_tilde+omega_hat) + f(omega_hat)
    # omega_tilde(1/dt+G/2) = -G/2*omega_hat + omega_hat/dt + f(omega_hat)
    # omega_tilde = inv(1/dt+G/2) * (-G/2*omega_hat + omega_hat/dt + f(omega_hat) ) 
    omega_tilde = (omega_hat/dt-G/2*omega_hat+nonlinear_terms(omega_hat)) / (1/dt + G/2)

    ## eq 3.5
    # (omega_hat_new - omega_hat)/dt = -G/2*(omega_hat_new+omega_hat) + 1/2(f(omega_tilde)+f(omega_hat))
    # omega_hat_new *(1/dt + G/2) = omega_hat/dt -G/2*omega_hat + 1/2 (f(omega_tilde)+f(omega_hat))
    # omega_hat_new = inv(1/dt + G/2) * omega_hat/dt -G/2*omega_hat + 1/2 (f(omega_tilde)+f(omega_hat))/
    omega_hat = (omega_hat/dt - G/2*omega_hat + 1/2*(nonlinear_terms(omega_hat)+nonlinear_terms(omega_tilde)) ) / (1/dt + G/2)
    return omega_hat

x = y = range(N)
model = Gaussian(dim = 2, var = 1, len_scale = 10)
srf = SRF(model,seed = 13,generator='Fourier',period = N)
field = srf.structured([x,y],seed=0)
w0 = torch.tensor(field) 
w0 = torch.zeros(n_traj,M,N).to(device)
for i in range(n_traj):
    s = 1000 if n_traj==1 else i
    w0[i,...] = torch.tensor((srf.structured([x, y], seed=s))).to(device) *ic_factor


omega_hat = torch.fft.fft2(w0)
# omega_hat2 = torch.fft.fft2(w0.unsqueeze(0))
# fig,axs = plt.subplots(1,2)

w_save = torch.zeros(n_traj,M,N,int(T/t_save))
with torch.no_grad():
    for step in tqdm(range(n_steps)):
        # omega_hat = euler_step(omega_hat, dt)
        omega_hat = update_step(omega_hat,nonlinear_terms,dt)
        # omega_hat2 = euler_step2(omega_hat2,dt)
        # save every t_save seconds
        if step % int(t_save/dt) == 0:
            w = torch.fft.ifft2(omega_hat).real
            # print(w.device)
            # w2 = torch.fft.ifft2(omega_hat2).real
            w_save[...,int(step*dt/t_save)] = w
            # # fig.clf()
            # axs[0].imshow(w.detach().cpu(), cmap='RdBu', origin='lower', extent=[0, domain_size, 0, domain_size])#,vmin=-25,vmax=25)
            # # axs[1].imshow(w2[0,...].detach().cpu(), cmap='RdBu', origin='lower', extent=[0, domain_size, 0, domain_size])#,vmin=-25,vmax=25)
            # plt.title(f"Time {step*dt}")
            # # plt.colorbar(label="Vorticity")
            # plt.pause(0.01)
torch.save(w_save,f'./data/{folder}/data.pt')

## pick 5 random to animate as a sanity check
print('animating...')
ind = np.random.randint(0,n_traj,n_ani)
fig,axs = plt.subplots(1,n_ani)

if n_ani == 1:
    axs = [axs]
ims = []
with torch.no_grad():
    for t in range(w_save.shape[-1]):
        frame_artists = []
        for i in range(n_ani):
            im = axs[i].imshow(w_save[ind[i], :, :, t].detach().cpu().numpy(),cmap='RdBu', animated=True)
            frame_artists.append(im)

        # Create a new text object every time
        time = t*t_save 
        title_obj = axs[i].text(0.5, 1.05, f"Time {time}", transform=axs[i].transAxes,
                                    ha='center', va='bottom', fontsize=12, animated=True)
        frame_artists.append(title_obj)

        ims.append(frame_artists)

ani = animation.ArtistAnimation(fig, ims, interval=1e-3)
ani.save(f"data/{folder}/data_ani.gif")


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





