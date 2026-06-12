import argparse
import os
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from model import ECO
from utils import (
    compare_distributions,
    evaluate_fourier_spectrum,
    pca_histogram_eval,
    plot_spatial_sum,
    visualize_ellipsoid,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _npz_value(value):
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        return value.tolist()
    return value


def load_model(model_dir):
    params_path = os.path.join(model_dir, "model_params.npz")
    ckpt_path = os.path.join(model_dir, "model_epoch_best.pt")

    params_npz = np.load(params_path, allow_pickle=True)
    model_params = {key: _npz_value(params_npz[key]) for key in params_npz.files}
    model_params['backbone'] = model_params.get('backbone','deeponet')
    if model_params['backbone'] == 'deeponet':
        TS = 0.05
    else:
        TS = 1.0
    model_params['trunk_scale']=model_params.get('trunk_scale',TS)

    model = ECO(model_params).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # Q_func initializes lazily in forward and the flag is not checkpointed.
    # Prevent eval from overwriting loaded Q_func weights.
    if model.project and model.nn_Q:
        model.V.Q_func.initialized_output = True

    model.eval()
    return model, model_params


def resample_grid_data(data, grid_size):
    u = data["u_batch"]
    x = data["x"]
    full_s = u.shape[-1]
    if full_s != grid_size:
        raise ValueError(f"Dataset grid is {full_s}, but requested grid is {grid_size}.")
    return u, x


def resample_state_to_size(state, target_size):
    if state.shape[-1] == target_size:
        return state
    return F.interpolate(
        state.unsqueeze(1),
        size=target_size,
        mode="linear",
        align_corners=True,
    ).squeeze(1)


@contextmanager
def temporarily_set_projection(model, enabled):
    old_value = model.project
    model.project = enabled
    try:
        yield
    finally:
        model.project = old_value


@contextmanager
def temporarily_set_latent_dim(model, latent_dim):
    if not model.project:
        yield
        return

    old_value = model.V.latent_dim
    model.V.latent_dim = latent_dim
    try:
        yield
    finally:
        model.V.latent_dim = old_value


def predict_step(model, state, eval_grid, train_grid, train_s, use_projection):
    if model.backbone == "deeponet":
        branch_state = resample_state_to_size(state, train_s)
        model_input = (branch_state, eval_grid)
    elif model.backbone == "fno":
        model_input = (state, eval_grid)
    else:
        raise ValueError(f"Unknown backbone: {model.backbone}")

    with temporarily_set_projection(model, False):
        base_pred = model(model_input)

    if not use_projection:
        return base_pred

    if model.nn_Q:
        with temporarily_set_latent_dim(model, eval_grid.shape[0]):
            return model.discrete_project((state, eval_grid), (base_pred, eval_grid))

    if state.shape[-1] != train_s or eval_grid.shape[0] != train_s:
        raise ValueError("Fixed Q/x0 projection can only be evaluated on the training discretization.")

    return model.discrete_project((state, eval_grid), (base_pred, eval_grid))


def rollout_model(model, gt_traj, eval_grid, train_grid, train_s, use_projection):
    gt_t = torch.tensor(gt_traj, dtype=torch.float32, device=device)
    pred = torch.zeros_like(gt_t)
    pred[:, 0, :] = gt_t[:, 0, :]

    with torch.no_grad():
        for t in range(gt_t.shape[1] - 1):
            pred[:, t + 1, :] = predict_step(
                model=model,
                state=pred[:, t, :],
                eval_grid=eval_grid,
                train_grid=train_grid,
                train_s=train_s,
                use_projection=use_projection,
            )
    return pred


def get_diag_Q_and_x0(model, grid, grid_size):
    if not model.project:
        return None, 30.0 ** 2, np.zeros((1, grid_size))

    c = model.c.detach().cpu().numpy() ** 2
    if model.V.nn_Q:
        Q_diag = model.V._construct_Q(grid).detach().cpu().numpy().reshape(-1)
        if model.V.nn_x0:
            x0 = model.V._construct_x0(grid).detach().cpu().numpy().reshape(1, grid_size)
        else:
            x0 = np.zeros((1, grid_size))
        return np.diag(Q_diag), c, x0

    if grid_size != model.V.latent_dim:
        return None, c, None

    Q = model.V._construct_Q().detach().cpu().numpy()
    if model.V.diag_Q:
        Q = np.diag(np.squeeze(Q))
    x0 = model.V.x_0.detach().cpu().numpy().reshape(1, grid_size)
    return Q, c, x0


def plot_v_history(model, gt_traj, pred_traj, grid, figs_dir, c, x0):
    if not model.project:
        gt = torch.tensor(gt_traj[0], dtype=torch.float32, device=device)
        pred = pred_traj[0]
        V_gt = torch.sum(gt ** 2, dim=-1)
        V_pred = torch.sum(pred ** 2, dim=-1)
    elif model.V.nn_Q:
        Q = model.V._construct_Q(grid).reshape(-1)
        if model.V.nn_x0:
            x0_t = model.V._construct_x0(grid).reshape(1, -1)
        else:
            x0_t = torch.zeros(1, grid.shape[0], device=device)
        gt = torch.tensor(gt_traj[0], dtype=torch.float32, device=device)
        pred = pred_traj[0]
        V_gt = torch.sum((gt - x0_t) ** 2 * Q, dim=-1)
        V_pred = torch.sum((pred - x0_t) ** 2 * Q, dim=-1)
    elif pred_traj.shape[-1] == model.V.latent_dim:
        Q = model.V._construct_Q()
        if model.V.diag_Q:
            Q = torch.diag(Q.reshape(-1))
        x0_t = model.V.x_0
        gt = torch.tensor(gt_traj[0], dtype=torch.float32, device=device)
        pred = pred_traj[0]
        V_gt = torch.einsum("ti,ij,tj->t", gt - x0_t, Q, gt - x0_t)
        V_pred = torch.einsum("ti,ij,tj->t", pred - x0_t, Q, pred - x0_t)
    else:
        print("Skipping V plot: fixed learned Q/x0 is not compatible with this grid size.")
        return

    plt.figure()
    plt.plot(V_pred.detach().cpu().numpy(), label="model")
    plt.plot(V_gt.detach().cpu().numpy(), label="GT")
    if model.project:
        plt.plot(np.array([0, len(V_gt)]), np.ones(2) * c, label="c")
    plt.xlabel("Time step")
    plt.ylabel("V")
    plt.yscale("log")
    plt.title("V over time")
    plt.legend()
    plt.savefig(os.path.join(figs_dir, "V_plot.png"))
    plt.close()


def plot_one_step(model, gt_traj, eval_grid, train_grid, train_s, use_projection, figs_dir):
    x_test = torch.tensor(gt_traj[0, :999, :], dtype=torch.float32, device=device)
    y_test = torch.tensor(gt_traj[0, 1:1000, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = predict_step(model, x_test, eval_grid, train_grid, train_s, use_projection)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].imshow(y_test.T.detach().cpu().numpy(), aspect="auto", vmin=-5, vmax=5)
    axs[0].set_title("Ground Truth")
    axs[0].set_xlabel("Time step")
    axs[0].set_ylabel("Position")

    im = axs[1].imshow(pred.T.detach().cpu().numpy(), aspect="auto", vmin=-5, vmax=5)
    axs[1].set_title("Model Prediction")
    axs[1].set_xlabel("Time step")
    axs[1].set_ylabel("Position")

    fig.tight_layout()
    fig.colorbar(im, ax=axs, location="right")
    plt.savefig(os.path.join(figs_dir, "1step.png"))
    plt.close(fig)


def plot_rollout_image(pred_traj, figs_dir):
    aspect = 0.5 * pred_traj.shape[1] / pred_traj.shape[2]
    plt.figure()
    plt.imshow(pred_traj[0].T.detach().cpu().numpy(), vmin=-5, vmax=5, aspect=aspect)
    plt.title("Rollout from Test Initial Condition")
    plt.colorbar()
    plt.xlabel("Time step")
    plt.ylabel("Position")
    plt.savefig(os.path.join(figs_dir, "rollout.png"))
    plt.close()


def plot_random_rollout(model, grid_size, eval_grid, train_grid, train_s, use_projection, figs_dir,
                        steps=10000, random_seed=10, random_IC_mag=20.0):
    torch.manual_seed(random_seed)
    u = torch.randn(1, grid_size, device=device) * random_IC_mag
    rollout = torch.zeros(steps, grid_size, device=device)

    with torch.no_grad():
        for t in range(steps):
            u = predict_step(model, u, eval_grid, train_grid, train_s, use_projection)
            rollout[t] = u[0]

    plt.figure()
    plt.imshow(rollout.T.detach().cpu().numpy(), aspect="auto", vmin=-5, vmax=5)
    plt.title("Rollout from Random Initial Condition")
    plt.colorbar()
    plt.xlabel("Time step")
    plt.ylabel("Position")
    plt.savefig(os.path.join(figs_dir, f"rollout_randomIC_mag{random_IC_mag}.png"))
    plt.close()


def select_data_for_grid(data_512, data_1024, grid_size):
    if grid_size == 1024:
        return data_1024
    return data_512


def run_eval_for_grid(model, model_params, data_512, data_1024, model_dir, grid_size, args):
    figs_dir = os.path.join(model_dir, f"eval_results_grid{grid_size}")
    os.makedirs(figs_dir, exist_ok=True)

    train_s = int(model_params["m"])
    data = select_data_for_grid(data_512, data_1024, grid_size)
    gt_u, x_np = resample_grid_data(data, grid_size)
    gt_traj = gt_u[:, ::5, :]

    eval_grid = torch.tensor(x_np, dtype=torch.float32, device=device).reshape(grid_size, 1)
    eval_grid = eval_grid * args.trunk_scale

    train_grid_data = select_data_for_grid(data_512, data_1024, train_s)
    train_grid_np = resample_grid_data(train_grid_data, train_s)[1]
    train_grid = torch.tensor(train_grid_np, dtype=torch.float32, device=device).reshape(train_s, 1)
    train_grid = train_grid * args.trunk_scale

    use_projection = bool(model.project and (grid_size == train_s or model.nn_Q))
    if model.project and not use_projection:
        print(
            f"Warning: disabling projection for grid {grid_size}. "
            "Fixed learned Q/x0 is not discretization invariant."
        )

    print(f"Evaluating grid {grid_size}; gt_traj shape: {gt_traj.shape}")
    plot_one_step(model, gt_traj, eval_grid, train_grid, train_s, use_projection, figs_dir)

    pred_traj = rollout_model(
        model=model,
        gt_traj=gt_traj,
        eval_grid=eval_grid,
        train_grid=train_grid,
        train_s=train_s,
        use_projection=use_projection,
    )
    print(f"Prediction shape for grid {grid_size}: {tuple(pred_traj.shape)}")

    Q, c, x0 = get_diag_Q_and_x0(model, eval_grid, grid_size)
    plot_v_history(model, gt_traj, pred_traj, eval_grid, figs_dir, c, x0)
    plot_rollout_image(pred_traj, figs_dir)
    plot_spatial_sum(gt_traj, pred_traj, figs_dir, traj_ind=0, save_name="spatial_sum.png")

    pred_np = pred_traj[0].detach().cpu().numpy()
    compare_distributions(
        gt_traj=gt_traj[0].ravel(),
        pred_traj=pred_np.ravel(),
        bins=50,
        plot=True,
        save_name=os.path.join(figs_dir, "distribution_traj.png"),
    )

    pca_gt, pca_pred = visualize_ellipsoid(
        gt_traj=gt_traj[0],
        test_traj=pred_traj[0],
        figs_dir=figs_dir,
        Q=Q,
        c=c,
        tag="",
    )
    pca_histogram_eval(
        gt_pca=pca_gt,
        pred_pca=pca_pred,
        bins=50,
        lim=[[-50.0, 50.0], [-50.0, 50.0]],
        save_path=os.path.join(figs_dir, "distribution_pca.png"),
        title_gt="Ground Truth",
        title_pred="Prediction",
    )

    evaluate_fourier_spectrum(
        gt_traj=gt_traj[0],
        star_traj=pred_np,
        save_path=os.path.join(figs_dir, "fourier_spectrum.png"),
    )

    plot_random_rollout(
        model=model,
        grid_size=grid_size,
        eval_grid=eval_grid,
        train_grid=train_grid,
        train_s=train_s,
        use_projection=use_projection,
        figs_dir=figs_dir,
        steps=args.random_rollout_steps,
        random_seed=args.random_seed,
        random_IC_mag=args.random_IC_mag,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing model_params.npz and model_epoch_best.pt.")
    parser.add_argument("--data_path", type=str, default="Data/KS_data_test_l100.53_grid512_M1_T2000.0_dt0.005_dt_sample0.2_amp20.0.npz/data.npz")
    parser.add_argument("--data_path_1024", type=str, default='Data/KS_data_test_l100.53_grid1024_M1_T2000.0_dt0.005_dt_sample0.2_amp20.0/data.npz')
    parser.add_argument("--trunk_scale", type=float, default=None)
    parser.add_argument("--random_rollout_steps", type=int, default=10000)
    parser.add_argument("--random_seed", type=int, default=10)
    parser.add_argument("--random_IC_mag", type=float, default=20.0)
    args = parser.parse_args()

    model, model_params = load_model(args.model_dir)
    args.trunk_scale = float(model_params["trunk_scale"])

    data_512 = np.load(args.data_path, allow_pickle=True)
    data_1024 = np.load(args.data_path_1024, allow_pickle=True)
    for grid_size in (512, 1024):
        run_eval_for_grid(model, model_params, data_512, data_1024, args.model_dir, grid_size, args)


if __name__ == "__main__":
    main()
