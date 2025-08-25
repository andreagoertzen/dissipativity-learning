import argparse
import numpy as np
from utils import gen_real_multi_traj, gen_multi_traj_scipy   # <--- adjust if your utils uses a different name
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def plot_traj3d(X_ds: np.ndarray, save_path: str, traj_idx: int = 0, stride: int = 1):
    """
    Plot a single 3D Lorenz-63 trajectory from X_ds with shape (N, T, 3).

    Args:
        X_ds: np.ndarray, shape (traj_num, traj_length, 3)
        traj_idx: which trajectory to plot [0 .. traj_num-1]
        stride: subsample factor along time for plotting density (>=1)
        title: optional plot title
        figsize: matplotlib figure size
    """
    if X_ds.ndim != 3 or X_ds.shape[2] != 3:
        raise ValueError(f"X_ds must have shape (N, T, 3); got {X_ds.shape}")

    N = X_ds.shape[0]
    if not (0 <= traj_idx < N):
        raise IndexError(f"traj_idx must be in [0, {N-1}]")

    xyz = X_ds[traj_idx]  # (T, 3)
    if stride > 1:
        xyz = xyz[::stride]

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, linewidth=1.0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Lorenz-63 trajectory #{traj_idx}")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"traj_{traj_idx}.png"))
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Generate Lorenz-63 dataset")
    parser.add_argument("--traj_num", type=int, default=10, help="Number of trajectories")
    parser.add_argument("--traj_length", type=int, default=20000, help="Trajectory length")
    parser.add_argument("--dt", type=float, default=0.001, help="Time step")
    parser.add_argument("--dt_target", type=float, default=0.05, help="Target time step for downsampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default="L63", help="Output file (.npy or .npz)")
    parser.add_argument("--xlim", type=float, default=50.0, help="Initial condition limit (+/-)")
    parser.add_argument("--integrator", type=str, default="rk4", help="Integrator to use")
    args = parser.parse_args()
    

    # Call your utils generator
    if args.integrator == "rk4":
        print("Using custom RK4 integrator")
        X_ds = gen_real_multi_traj(
            M=args.traj_num,
            N=args.traj_length,
            dt=args.dt,
            dt_target=args.dt_target,
            seed=args.seed,
            x_lim=args.xlim
        )
    elif args.integrator == "rk45":
        print("Using scipy RK45 integrator")
        # Use the new scipy-based generator
        X_ds = gen_multi_traj_scipy(
            M=args.traj_num,
            N=args.traj_length,
            dt_target=args.dt_target,
            seed=args.seed,
            x_lim=args.xlim
        )

     # Ensure array is numpy
    X_ds = np.asarray(X_ds, dtype=np.float32)

    # Some utils return a dict {"X_ds": array}, handle both cases
    if isinstance(X_ds, dict) and "X_ds" in X_ds:
        X_ds = X_ds["X_ds"]
   
    # Save
    save_path = "Data"
    save_path = os.path.join(save_path, args.out + f"_M{args.traj_num}_N{args.traj_length}_dt_s{args.dt_target}_dt{args.dt}_ic{args.xlim}_int{args.integrator}")

    os.makedirs(save_path, exist_ok=True)
    save_name = "data.npz"
    np.savez(os.path.join(save_path, save_name), X_ds=X_ds)
    print(f"Saved dataset to {os.path.join(save_path, save_name)}, shape={X_ds.shape}")
    
    for i in range(args.traj_num):
        plot_traj3d(X_ds, save_path=save_path, traj_idx=i)


if __name__ == "__main__":
    main()