import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from matplotlib.patches import Ellipse
from model import stable_lorenz_model
from utils import tensor_lorenz96_5, gen_real_multi_traj, tensor_r_KSNASA
from tqdm import tqdm

import pickle
import os
import argparse

from scipy.fft import ifft as sp_ifft
from scipy.spatial.distance import jensenshannon
from scipy.linalg import sqrtm

from sklearn.decomposition import PCA

def load_L96_traj(data_loc):
    with open(data_loc, 'rb') as f:
        test_results = pickle.load(f)

    GT_traj = test_results['Test_trajs']['GT'].detach().cpu().numpy()
    Star_traj = test_results['Test_trajs']['traj_1'].detach().cpu().numpy()
    Hat_traj = test_results['Test_trajs']['traj_2'].detach().cpu().numpy()

    return GT_traj, Star_traj, Hat_traj

def load_L96_model_Q(model_loc):
	# Load the dictionary, ensuring it's mapped to the CPU for compatibility
    state_dict = torch.load(model_loc, map_location=torch.device('cpu'))

    # Access the tensors directly using their saved keys
    L = state_dict['V.L']
    c_tensor = state_dict['c']
    x_0_tensor = state_dict['V.x_0']
    
    # --- The rest of the logic is the same as before ---
    
    # Ensure L is lower triangular with positive diagonal elements
    L = torch.tril(L)
    L.diagonal().exp_()
    
    # Calculate the positive definite matrix Q
    Q = L @ L.t()
    
    # Convert tensors to NumPy arrays for further processing
    Q_np = Q.detach().numpy()
    c_np = (c_tensor**2).detach().numpy()
    x_0_np = x_0_tensor.detach().numpy()
 
    # print("Successfully extracted Q, c, and x_0.")
    return Q_np, c_np, x_0_np


def GT_based_PCA_eval(gt_traj, pred_traj, PCA_n=2, save_path=None):
    pca = PCA(n_components=PCA_n)
    gt_pca = pca.fit_transform(gt_traj)
    pred_pca = pca.transform(pred_traj)

    plt.figure(figsize=(10, 10))
    plt.scatter(gt_pca[:, 0], gt_pca[:, 1], s=2, alpha=0.5, label='GT')
    plt.scatter(pred_pca[:, 0], pred_pca[:, 1], s=1, alpha=0.5, label='f_hat')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    plt.tight_layout()
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.close()

    U = pca.components_  # U has shape (2, state_dim)
    return U, gt_pca, pred_pca, pca

def plot_projected_point_cloud_bound(Q, c, x0, pca, pca_gt_traj, pca_star_traj, save_path=None):
    """
    Visualizes the boundary of a high-dimensional ellipsoid by projecting its
    surface points into the 2D PCA space and plotting them as a point cloud.

    Args:
        Q (np.ndarray): The (D, D) matrix of the quadratic Lyapunov function.
        c (float): The level set value.
        x0 (np.ndarray): The (D,) center of the ellipsoid.
        pca (PCA): The fitted scikit-learn PCA object.
        pca_gt_traj (np.ndarray): The (N, 2) projected ground truth trajectory.
        pca_star_traj (np.ndarray): The (N, 2) projected model trajectory.
        save_path (str, optional): Path to save the figure.
    """
    print("Generating projected point cloud of the ellipsoid boundary...")
    D = Q.shape[0]
    
    if x0.ndim == 2:
        x0 = x0.flatten()

    # 1. Generate a large number of points on a unit sphere in D dimensions
    num_points = int(5e5)  # More points create a denser, clearer boundary
    z = np.random.randn(D, num_points)
    z /= np.linalg.norm(z, axis=0)

    # 2. Map these points to the surface of the 32D ellipsoid
    try:
        # Use (x - x0)^T Q (x - x0) = c => x = x0 + inv(sqrt(Q)) @ z * sqrt(c)
        Q_inv_sqrt = np.linalg.inv(sqrtm(Q))
    except np.linalg.LinAlgError:
        print("Warning: The learned matrix Q is singular. Using pseudo-inverse.")
        Q_inv_sqrt = np.linalg.pinv(sqrtm(Q))
        
    ellipsoid_surface_pts = x0[:, np.newaxis] + Q_inv_sqrt @ z * np.sqrt(c)

    # 3. Project these 32D surface points down to the 2D PCA space
    projected_boundary_pts = pca.transform(ellipsoid_surface_pts.T)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 4. Plot the projected boundary as a scatter plot (a point cloud)
    ax.scatter(projected_boundary_pts[:, 0], projected_boundary_pts[:, 1], 
               s=1,              # Small point size
               c='red', 
               alpha=0.2,        # Low opacity for a "cloud" effect
               label='Projected Ellipsoid',
               zorder=1)         # Plot behind the trajectories

    # Plot the trajectories on top
    ax.scatter(pca_gt_traj[:, 0], pca_gt_traj[:, 1], s=5, alpha=0.5, label="Ground Truth", zorder=2)
    ax.scatter(pca_star_traj[:, 0], pca_star_traj[:, 1], s=5, alpha=0.5, label="f^* (Projection)", zorder=3)

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('Projected Trajectories and Ellipsoid Boundary')
    ax.legend()
    ax.axis('equal')

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to {save_path}")

    plt.close(fig)
    
def filter_nan_traj(Hat_traj):
    # This is assuming the Hat_traj already has (num_points, point_dim) shape
    valid_mask = ~np.isnan(Hat_traj).any(axis=1)
    Hat_valid = Hat_traj[valid_mask]
    return Hat_valid

def hists_to_KL(xedges, yedges, bins, H1, H2, eps=1e-9, option='KL'):
    bin_width_x = (xedges[-1] - xedges[0]) / bins
    bin_width_y = (yedges[-1] - yedges[0]) / bins
    bin_area = bin_width_x * bin_width_y

    H1 = H1 * bin_area
    H1 = H1 / H1.sum()
    H2 = H2 * bin_area
    H2 = H2 / H2.sum()

    h1 = H1.ravel()
    h2 = H2.ravel()

    h1 = h1 + eps
    h2 = h2 + eps
    h1 /= h1.sum()
    h2 /= h2.sum()
    
    if option == 'KL':
        result = np.sum(h1 * np.log(h1 / h2))
    elif option == 'JS':
        result = jensenshannon(h1, h2)

    return result

def pca_histogram_eval(gt_pca, pred_pca, bins=50, lim=[[-10.0, 20.0], [-10.0, 20.0]], save_path=None, title_gt='GT', title_pred='Pred'):
    # ----- histogram -----
    if lim is None:          # auto-range
        mx = np.max(np.abs(np.vstack((gt_pca, pred_pca))), axis=0)
        lim = [[-mx[0], mx[0]], [-mx[1], mx[1]]]

    H_gt,  xe, ye = np.histogram2d(gt_pca[:,0],  gt_pca[:,1],
                                   bins=bins, range=lim, density=True)
    
    H_pr,  _,  _  = np.histogram2d(pred_pca[:,0], pred_pca[:,1],
                                   bins=bins, range=lim, density=True)

    # ----- KL / JS -----
    KL = hists_to_KL(xe, ye, bins, H_gt, H_pr, option='KL')
    JS = hists_to_KL(xe, ye, bins, H_gt, H_pr, option='JS')

    # ----- figure -----
    if save_path:
        vmax = max(H_gt.max(), H_pr.max())

        fig, ax = plt.subplots(1, 2, figsize=(12,6))
        for A, H, ttl in zip(ax, (H_gt, H_pr), (title_gt, title_pred)):
            im = A.imshow(H.T, origin='lower', extent=[xe[0],xe[-1],ye[0],ye[-1]],
                          vmin=0, vmax=vmax, aspect='auto')
            A.set_title(ttl)

        plt.tight_layout()
        
        # Make room on the right
        fig.subplots_adjust(right=0.85)

        # Create a new Axes on the right side of the figure for the colorbar
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  
        # [left, bottom, width, height] in figure coordinates

        # Now draw the colorbar on that new Axes
        cbar = fig.colorbar(im, cax=cbar_ax)
        # cbar.set_label("Density")  # Optional label

        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    return KL, JS

if __name__ == "__main__":
    # L96 save directory
    save_dir = "PNAS_PCA"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    print('-'*32)
    print("Starting L96 evaluation")
    L96_save_dir = os.path.join(save_dir, "L96")
    if not os.path.exists(L96_save_dir):
        os.makedirs(L96_save_dir, exist_ok=True)

    model_loc = 'Trained_Models/L96_5_proj/models/E29999.pt'
    Q, c, x_0 = load_L96_model_Q(model_loc)
    
    data_loc = 'L96_eval/L96_test_20_20000.pkl'
    gt_traj, Star_traj, Hat_traj = load_L96_traj(data_loc)
    
    # flatten the trajectories
    # gt_flat = gt_traj.reshape(-1, gt_traj.shape[-1])
    # Star_flat = Star_traj.reshape(-1, Star_traj.shape[-1])
    gt_flat = gt_traj[0]
    Star_flat = Star_traj[0]

    U, gt_pca, pred_pca, pca = GT_based_PCA_eval(gt_flat, Star_flat, PCA_n=2)
    plot_projected_point_cloud_bound(Q, c, x_0, pca, gt_pca, pred_pca, save_path=os.path.join(L96_save_dir, "fstar_PCA_scatter.png"))
    KL, JS = pca_histogram_eval(gt_pca, pred_pca, bins=50, lim=[[-10.0, 20.0], [-10.0, 20.0]], save_path=os.path.join(L96_save_dir, "fstar_PCA_hist.png"), title_gt='GT', title_pred='f^* Projected')
    print(f"KL Divergence: {KL}, Jensen-Shannon Divergence: {JS}")
    print("Finish L96 evaluation")
    print('-'*32)
    
    print("Starting KS evaluation")
    KS_save_dir = os.path.join(save_dir, "KS_ROM")
    if not os.path.exists(KS_save_dir):
        os.makedirs(KS_save_dir, exist_ok=True)

    model_loc = 'Trained_Models/KS_proj/model.pt'
    Q, c, x_0 = load_L96_model_Q(model_loc)
    
    data_loc = 'L96_eval/L96_test_20_20000.pkl'
    gt_traj, Star_traj, Hat_traj = load_L96_traj(data_loc)
    
    