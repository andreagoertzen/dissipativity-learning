import torch
import numpy as np
import scipy.io


#################################################
#
# w_to_u function from: https://github.com/neuraloperator/markov_neural_operator/blob/main/utilities.py
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def w_to_u(w):
    batchsize = w.size(0)
    nx = w.size(1)
    ny = w.size(2)

    device = w.device
    w = w.reshape(batchsize, nx, ny, -1)

    w_h = torch.fft.fft2(w, dim=[1, 2])
    # Wavenumbers in y-direction
    k_max = nx // 2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(N, 1).repeat(1,
                                                                                                        N).reshape(
        1, N, N, 1)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, N).repeat(N,
                                                                                                        1).reshape(
        1, N, N, 1)
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[0, 0, 0, 0] = 1.0
    f_h = w_h / lap

    ux_h = 1j * k_y * f_h
    uy_h = -1j * k_x * f_h

    ux = torch.fft.irfft2(ux_h[:, :, :k_max + 1], dim=[1, 2])
    uy = torch.fft.irfft2(uy_h[:, :, :k_max + 1], dim=[1, 2])
    u = torch.cat([ux, uy], dim=-1)
    return u

