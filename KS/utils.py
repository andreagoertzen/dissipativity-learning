import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from tqdm import tqdm
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.special import rel_entr
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon


class TrajectoryDataset(Dataset):
    """
    Custom PyTorch Dataset for DeepONet trajectory data.
    
    This class takes trajectory data of shape (num_traj, traj_length, traj_dim)
    and creates input-output pairs of (u_t, u_{t+1}).
    """
    def __init__(self, u_data, x_data):
        """
        Args:
            u_data (np.array): Trajectory data with shape (num_traj, traj_length, traj_dim).
            x_data (np.array): Constant input for the Trunk network.
        """
        super().__init__()
        # The trunk input is constant for all samples, so convert it once.
        if x_data.ndim == 1:
            x_data = x_data.reshape(-1, 1)

        self.trunk_input = torch.tensor(x_data, dtype=torch.float32)
        
        branch_inputs = []
        targets = []
        
        # Iterate over each trajectory to create (u_t, u_{t+1}) pairs
        for traj in u_data:
            # A trajectory of length L has L-1 possible pairs
            for t in range(traj.shape[0] - 1):
                branch_inputs.append(traj[t])
                targets.append(traj[t+1])
                
        # Convert the lists of individual steps into single large tensors for efficiency.
        self.branch_inputs = torch.tensor(np.array(branch_inputs), dtype=torch.float32)
        self.targets = torch.tensor(np.array(targets), dtype=torch.float32)
        
    def __len__(self):
        """Returns the total number of (u_t, u_{t+1}) pairs."""
        return self.branch_inputs.shape[0]
    
    def __getitem__(self, idx):
        """
        Fetches the sample at the given index.
        
        Returns:
            tuple: A tuple containing ((branch_input, trunk_input), target).
                   This format is convenient for unpacking during the training loop.
        """
        branch_in = self.branch_inputs[idx]
        target = self.targets[idx]
        
        model_input = (branch_in, self.trunk_input)
        
        return model_input, target
    
def load_multi_traj_data(data,trunk_scale):
    u_all_traj = data['u_batch']
    x_trunk_input = data['x']*trunk_scale
    num_traj = u_all_traj.shape[0]

    # Split trajectories into training and validation sets (80/20)
    # It's important to split trajectories, not individual time-steps,
    # to prevent data leakage between train and validation.
    num_train_traj = int(0.8 * num_traj)

    u_train_traj = u_all_traj[:num_train_traj]
    u_val_traj = u_all_traj[num_train_traj:]

    train_dataset = TrajectoryDataset(u_data=u_train_traj, x_data=x_trunk_input)
    val_dataset = TrajectoryDataset(u_data=u_val_traj, x_data=x_trunk_input)

    return train_dataset, val_dataset

def val_onestep_visual(model, data, device, figs_dir='figs'):
	"""
	Generates and saves a plot comparing the one-step model prediction
	against the ground truth for an entire trajectory.

	Args:
		model (torch.nn.Module): The trained DeepONet model.
		x_test (tuple): A tuple containing branch and trunk inputs for the test set.
		y_test (torch.Tensor): The ground truth target values for the test set.
		figs_dir (str): The directory where the output plot will be saved.
	"""
	print("Generating one-step prediction plot...")
	model.eval()

	num_train_traj = int(0.8 * data['u_batch'].shape[0])
	traj_val = data['u_batch'][num_train_traj:, :, :]
	x_trunk_input = torch.tensor(data['x'], dtype=torch.float32).to(device)
	x_trunk_input = x_trunk_input.unsqueeze(1)

	num_val_traj = traj_val.shape[0]

	for i in range(num_val_traj):
		x_test = (torch.tensor(traj_val[i, :, :], dtype=torch.float32).to(device), x_trunk_input)
		with torch.no_grad():
			u_test_pred = model(x_test)

		fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
		
		# Plot Ground Truth
		im1 = axs[0].imshow(
			traj_val[i, :, :].T,
			aspect='auto', vmin=-2.5, vmax=2.5, cmap='viridis'
		)
		axs[0].set_title('Ground Truth')
		axs[0].set_ylabel('Position')

		# Plot Model Prediction
		im2 = axs[1].imshow(
			u_test_pred.T.detach().cpu().numpy(),
			aspect='auto', vmin=-2.5, vmax=2.5, cmap='viridis'
		)
		axs[1].set_title('Model One-Step Prediction')
		axs[1].set_xlabel('Time (steps)')
		axs[1].set_ylabel('Position')

		fig.tight_layout()
		fig.colorbar(im2, ax=axs, location='right', label='Value')
		plt.savefig(f'{figs_dir}/1_one_step_prediction_traj{i}.png', dpi=300)
		plt.close(fig)

def run_model_visualization(
    model,
    x_test,
    y_test,
    s,
    device,
    figs_dir='figs2',
    figs_tag = '',
    rollout_steps_test=1000,
    rollout_steps_random=10000,
    random_seed=10,
    random_IC_mag=1.0
):

    # --- 1. One-step prediction visualization ---
    with torch.no_grad():
        u_test = model(x_test)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    # print("y_test shape:", y_test.shape)
    # print("u_test shape:", u_test.shape)

    # extent = [physical_t[90000] * dt, physical_t[-1] * dt, physical_x[0], physical_x[-1]]

    axs[0].imshow(
        y_test.T.detach().cpu().numpy().astype(np.float32),
        #extent=extent,
        aspect='auto',
        vmin=-5,
        vmax=5,
    )
    axs[0].set_title('Ground Truth')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Position')

    im = axs[1].imshow(
        u_test.T.detach().cpu().numpy().astype(np.float32),
        #extent=extent,
        aspect='auto',
        vmin=-5,
        vmax=5,
    )
    axs[1].set_title('Model Prediction')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Position')

    fig.tight_layout()
    fig.colorbar(im, ax=axs, location='right')
    plt.savefig(f'{figs_dir}/1step.png')
    plt.close(fig)

    # # --- 2. Long trajectory rollout (on test data) ---
    # rollout_traj = torch.zeros(rollout_steps_test, s)
    # u_out = x_test[0][0, ...]  # initial condition
    # print(u_out.shape)
    # u_out = u_out.unsqueeze(0)

    # for i in range(rollout_steps_test):
    #     u_out = model((u_out, x_test[1]))
    #     rollout_traj[i, :] = u_out

    # plt.figure()
    # plt.imshow(
    #     rollout_traj.T.detach().numpy().astype(np.float32),
    #     extent=[0, rollout_steps_test, 0, s],
    #     aspect='auto',
    #     vmin=-5, vmax=5
    # )
    # plt.title('Rollout on Test Data')
    # plt.colorbar()
    # plt.savefig(f'{figs_dir}/rollout_test.png')
    # plt.close()

    # --- 3. Rollout from random initial condition ---

    # use a pytorch seed
    torch.manual_seed(random_seed)
    u0 = torch.randn(1, s).to(device)

    print("Random IC shape:", u0.shape)

    rollout_traj = torch.zeros(rollout_steps_random, s)
    u_out = u0*random_IC_mag

    with torch.no_grad():
        for i in range(rollout_steps_random):
            u_out = model((u_out, x_test[1]))
            rollout_traj[i, :] = u_out

    plt.figure()
    plt.imshow(
        rollout_traj.T.detach().numpy().astype(np.float32),
        extent=[0, rollout_steps_random, 0, s],
        aspect='auto',
        vmin=-5,vmax=5
    )
    plt.title('Rollout from Random Initial Condition')
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.savefig(f'{figs_dir}/rollout_randomIC_mag{random_IC_mag}.png')
    plt.close()


def visualize_ellipsoid(gt_traj, test_traj, figs_dir, Q=None, c=1.0,tag=''):
    # Perform a PCA on the reshaped data, data is of size (num_traj, traj_length, traj_dim), we can lump all trajectory together into (traj_length, traj_dim)
    reshaped_data = gt_traj.reshape(-1, gt_traj.shape[-1])
    if test_traj is not None:
        pred_traj = test_traj.reshape(-1, test_traj.shape[-1]).detach().cpu().numpy()

    print(reshaped_data.shape)
    pca = PCA(n_components=2)
    pca_traj_1 = pca.fit_transform(reshaped_data)
    if test_traj is not None:
        # pca_traj_pred = pca.fit_transform(pred_traj)
        pca_traj_pred = pca.transform(pred_traj)
    U = pca.components_

    if Q is None:
        # If no Q is provided, use the covariance
        Q = np.eye(U.shape[-1])
        c = c ** 2
        
    Q_inv = np.linalg.inv(Q)
    A = np.linalg.inv(U @ Q_inv @ U.T)

    # ----------------------------
    # Extract ellipse parameters from A:
    # The ellipse in PCA space is given by y^T A y = c.
    # Its semi-axis lengths are given by sqrt(c / eigenvalue).
    # The eigenvectors determine the orientation.
    # ----------------------------
    eigvals, eigvecs = np.linalg.eigh(A)
    print(eigvals)
    # Sort eigenvalues and eigenvectors in descending order.
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Compute semi-axis lengths (for y^T A y = c)
    # print(eigvals)
    axis_length1 = np.sqrt(c / eigvals[0])
    axis_length2 = np.sqrt(c / eigvals[1])

    # Compute the rotation angle (in degrees) of the ellipse.
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    # ----------------------------
    # Plot the projected ellipsoid in the 2D PCA space.
    # ----------------------------
    fig, ax = plt.subplots(figsize=(10, 10))

    # For a centered ellipsoid, the projected center is U*(x0 - x0)=0.
    ellipse_patch = Ellipse(xy=(0, 0),
                            width=2 * axis_length1,   # full axis length in the first direction
                            height=2 * axis_length2,  # full axis length in the second direction
                            angle=angle,
                            edgecolor='red', facecolor='none', lw=2
                            )
    ax.add_patch(ellipse_patch)

    # Optionally, plot the PCA-transformed trajectory data for context.
    ax.scatter(pca_traj_1[:, 0], pca_traj_1[:, 1], s=2, alpha=0.3, label="PCA GT")
    if test_traj is not None:
        ax.scatter(pca_traj_pred[:, 0], pca_traj_pred[:, 1], s=2, alpha=0.3, label="PCA Pred")
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)

    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend()
    plt.axis('equal')
    plt.savefig(f'{figs_dir}/PCA_ellipsoid_{tag}.png')
    plt.close()
    return pca_traj_1[:,:2], pca_traj_pred[:,:2]

def rollout_on_test(eval_model, data_x, trunk_scale, test_traj, device, figs_dir, project,c):
    eval_model.eval()
    trunk_input = torch.tensor(data_x, dtype=torch.float32).to(device) * trunk_scale
    # make trunk input 512 by 1, now it's 512
    trunk_input = trunk_input.view(512, 1)

    # print("trunk_input shape:", trunk_input.shape)
    test_traj = torch.tensor(test_traj, dtype=torch.float32).to(device)
    pred_traj = torch.zeros_like(test_traj).to(device)
    pred_traj[:, 0, :] = test_traj[:, 0, :].to(device)

    dt = 1
    # Q = eval_model.V._construct_Q().to(device)
    V_hist = torch.zeros(test_traj.shape[1]-1).to(device)
    V_hist_GT = torch.zeros(test_traj.shape[1]-1).to(device)

    if not project:
        c = c
    else:
        c = eval_model.c 

    for t in tqdm(range(test_traj.shape[1]-1)):
    # for t in tqdm(range(150)):
        with torch.no_grad():
        # Forward pass through the model
            input_t = (pred_traj[:, t, :], trunk_input)
            pred_traj[:, t+1, :] = eval_model(input_t)

            w_in = pred_traj[0, t, :]
            w_out = pred_traj[0, t+1, :]
            # w_diff = w_in - eval_model.V.x_0

            # dVdw = 2 * (w_diff @ Q)
            # # cond = (dVdw * w_out).sum(dim=1) - (dVdw * w_in).sum(dim=1) + dt * (eval_model.V(w_in) - eval_model.c ** 2)
            # A = dVdw
            # bx = eval_model.V(w_in)-(1/dt) * torch.einsum('bi,bi->b',dVdw, w_in) - eval_model.c**2
            # cond = torch.einsum('bi,bi->b',A,w_out) + bx
            # # print((dVdw ** 2).sum(dim=1))

            # if cond > 0:
            # 	print(f"Condition violated at timestep {t}: {cond.item()}")
            # # print(cond)

            if not project:
                # If no Q is provided, use the covariance
                Q = torch.eye(trunk_input.shape[0]).to(device)

                V_hist[t] = w_in @ Q @ w_in.T #torch.einsum('bi,ij,bj->b', w_in, Q, w_in)
                V_in = w_in @ Q @ w_in.T #torch.einsum('bi,ij,bj->b', w_in, Q, w_in)
                V_out = w_out @ Q @ w_out.T #torch.einsum('bi,ij,bj->b', w_out, Q, w_out)
                V_hist_GT[t] = test_traj[:,t,:] @ Q @ test_traj[:,t,:].T
            else:
                Q = eval_model.V._construct_Q()
                V_hist[t] = eval_model.V(w_in)
                V_in = eval_model.V(w_in)
                V_out = eval_model.V(w_out)
                w0 = eval_model.V.x_0
                V_hist_GT[t] = (test_traj[:,t,:]-w0) @ Q @ (test_traj[:,t,:]-w0).T

            
            if V_in > c ** 2:
                cond = V_out - V_in
                if cond > 0:
                    print(f"Condition violated at timestep {t} OUTSIDE: {cond.item()}")
            else:
                cond = V_out - c ** 2
                if cond > 0:
                    print(f"Condition violated at timestep {t} INSIDE: {cond.item()}")

    # plot the V_hist against time
    plt.plot(V_hist.cpu().numpy(),label='model')
    plt.plot(V_hist_GT.cpu().numpy(),label='GT')

    # plot c as a single line
    # plt.plot(eval_model.c.detach().cpu().numpy() ** 2, label='c')
    plt.xlabel("Time step")
    plt.ylabel("V")
    plt.yscale("log")
    plt.title("V over time")
    plt.legend()
    plt.savefig(f'{figs_dir}/V_plot.png')
    plt.close()

    # plt.figure()
    # plt.imshow(pred_traj[0,...].T.cpu().numpy(),aspect="auto")
    # plt.savefig(f'{figs_dir}/traj_forPCA.png')

    return pred_traj



def compare_distributions(gt_traj, pred_traj, bins=50, plot=True, save_name='distribution.png'):
    """
    Compute KL divergence between two histograms (discrete distributions).
    """
    # Combine to get shared bin edges
    all_data = np.concatenate([gt_traj, pred_traj])
    bin_edges = np.linspace(np.min(all_data), np.max(all_data), bins + 1)

    # Histogram counts
    p_counts, _ = np.histogram(gt_traj, bins=bin_edges)
    q_counts, _ = np.histogram(pred_traj, bins=bin_edges)

    # Convert to probabilities
    p = p_counts / np.sum(p_counts)
    q = q_counts / np.sum(q_counts)

    # Clip to avoid log(0)
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)

    # Discrete KL divergence
    kl_div = np.sum(rel_entr(p, q))

    # Plot with seaborn
    if plot:
        plt.figure(figsize=(8, 5))
        sns.kdeplot(gt_traj, label='Ground Truth', fill=False, alpha=0.7, color='blue')
        sns.kdeplot(pred_traj, label='Model Prediction', fill=False, alpha=0.7, color='red')
        plt.title(f'Comparison (KL Divergence = {kl_div:.4f})')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_name)
        plt.close()

    return kl_div


def pca_histogram_eval(gt_pca, pred_pca, bins=50, lim=[[-50.0, 50.0], [-50.0, 50.0]], save_path=None, title_gt='GT', title_pred='Projection'):
    # ----- histogram -----
    if lim is None:          # auto-range
        mx = np.max(np.abs(np.vstack((gt_pca, pred_pca))), axis=0)
        lim = [[-mx[0], mx[0]], [-mx[1], mx[1]]]

    H_gt,  xe, ye = np.histogram2d(gt_pca[:,0],  gt_pca[:,1],
                                   bins=bins, range=lim, density=True)
    
    H_pr,  _,  _  = np.histogram2d(pred_pca[:,0], pred_pca[:,1],
                                   bins=bins, range=lim, density=True)

    # ----- KL / JS -----
    KL_pred = hists_to_KL(xe, ye, bins, H_gt, H_pr, option='KL')
    JS_pred = hists_to_KL(xe, ye, bins, H_gt, H_pr, option='JS')

    # ----- figure -----
    if save_path:
        vmax = max(H_gt.max(), H_pr.max())

        fig, ax = plt.subplots(1, 2, figsize=(18,6))
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
    return KL_pred, JS_pred 

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



def evaluate_fourier_spectrum(gt_traj, star_traj, save_path=None):
    """
    Computes, plots, and evaluates the Fourier power spectra of trajectories.
    Handles both single (T, D) and batched (B, T, D) trajectory inputs.
    """
    # Helper function to calculate the average power spectrum from a single trajectory
    def get_power_spectrum(traj):
        fft_coeffs = np.fft.fft(traj, axis=0)
        power = np.abs(fft_coeffs)**2
        avg_power = np.mean(power, axis=1)
        return avg_power[:len(avg_power) // 2]

    # Helper function to calculate Mean Squared Error on a log scale
    def calculate_spectrum_error(spec_true, spec_pred, eps=1e-12):
        log_spec_true = np.log10(spec_true + eps)
        log_spec_pred = np.log10(spec_pred + eps)
        return np.mean((log_spec_true - log_spec_pred)**2)

    # Check if the input is batched by checking the number of dimensions
    if gt_traj.ndim == 3:
        batch_size = gt_traj.shape[0]
        print(f"\n--- Performing Batched Fourier Evaluation ({batch_size} trajectories) ---")
        
        batch_errors_star = []
        
        # Store spectra from each batch item to average them later for plotting
        all_spec_gt = []
        all_spec_star = []

        for i in range(batch_size):
            # Get single trajectories from the batch
            gt_single = gt_traj[i]
            star_single = star_traj[i]

            # Calculate spectrum for this item
            spec_gt = get_power_spectrum(gt_single)
            spec_star = get_power_spectrum(star_single)

            # Store spectra for averaging
            all_spec_gt.append(spec_gt)
            all_spec_star.append(spec_star)
            
            # Calculate and store errors for this item
            batch_errors_star.append(calculate_spectrum_error(spec_gt, spec_star))

        # Average the errors and spectra across the entire batch
        final_error_star = np.mean(batch_errors_star)
        
        plot_spec_gt = np.mean(all_spec_gt, axis=0)
        plot_spec_star = np.mean(all_spec_star, axis=0)

    elif gt_traj.ndim == 2:
        print("\n--- Performing Single Trajectory Fourier Evaluation ---")
        # For a single trajectory, the final error is just the calculated error
        plot_spec_gt = get_power_spectrum(gt_traj)
        plot_spec_star = get_power_spectrum(star_traj)
        
        final_error_star = calculate_spectrum_error(plot_spec_gt, plot_spec_star)
    else:
        raise ValueError(f"Input trajectory has an unsupported shape: {gt_traj.shape}. Expected 2 or 3 dimensions.")

    # --- Common code for printing and plotting ---
    
    print(f"Spectrum Difference (GT vs. f* Projected): {final_error_star:.6f}")
    print("---------------------------------")
    
    if save_path:
        plt.figure(figsize=(12, 7))
        modes = np.arange(len(plot_spec_gt))

        plt.plot(modes, plot_spec_gt, label='Ground Truth', color='black', linewidth=2.5, alpha=0.8)
        plt.plot(modes, plot_spec_star, label=r'$f^*$ Projected', color='#1f77b4', linestyle='--')

        plt.yscale('log')
        plt.xlabel('Frequency Mode Index')
        plt.ylabel('Power (Log Scale)')
        plt.title('Fourier Power Spectrum Comparison')
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved Fourier spectrum plot to {save_path}")

    return final_error_star