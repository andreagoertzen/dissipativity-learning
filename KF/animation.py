# animation.py
import torch, os
from pathlib import Path
from utils import animate_saved_rollout

rollout_path = Path("FNO_C2000_rollout.pt").resolve()  # absolute path
figs_dir = rollout_path.parent                           # a real directory

# load GT exactly like eval.py
data_animate = torch.load("data/KF_Re500_M128_tsave0.5_T5000_n1/data.pt")[..., ::2]
s = data_animate.shape[1]
data_animate = data_animate[..., :500].permute(0, 3, 1, 2).reshape(-1, s * s)
gt_seq = data_animate[1:, ...]

# make sure dir exists
figs_dir.mkdir(parents=True, exist_ok=True)

animate_saved_rollout(
    pred_traj_path=str(rollout_path),
    gt_seq=gt_seq,
    figs_dir=str(figs_dir),   # not empty now
    s=s,
    out_name="rollout.gif",
    max_frames=500,           # optional
    stride=1,
    fps=5,
)
