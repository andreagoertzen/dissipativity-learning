import numpy as np

cs = [100.0,125.0]
for c in cs:
    model_params = {
        'm': 64,
        'n': 2,
        'trainable_c': False,
        'c0': c,
        'project': True,
        'diag_Q': True,
        'branch_conv_channels': [64,128,256,512],
        'branch_fc_dims': [1024],
        'trunk_hidden_dims': [1024,1024,1024,1024],
        'output_dim': 1024,
        'dt': 1.0,
        'discrete_proj': True,
    }
    np.savez(f"./Trained_Models/0912_22/E5000_Re40_TS1.0_branchConv4_trunkHidden4_dt1.0_lr2e-05_bsize2048__proj_LamRegVol0.1_C0{c}_diagQdiscreteProj_dim1024/model_params.npz", **model_params)
