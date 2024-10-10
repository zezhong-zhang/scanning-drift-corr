#%%
from orth_scan import OrthScan
import numpy as np
import matplotlib.pyplot as plt
import h5py

with h5py.File('/home/zzhang/OneDrive/code/scanning-drift-corr/data_examples/nonlinear_drift_correction_synthetic_dataset_for_testing.mat', 'r') as f:
    img1 = f['image00deg'][:].T
    img2 = f['image90deg'][:].T
    img_gt = f['imageIdeal'][:]


osc = OrthScan([img1, img2], scan_angles=[0, 90], low_pass_sigma=32, report_progress=True, padding_scale=1.5, kde_sigma=0.5, edge_width=1/128, linear_search_steps=2)
osc.run_linear_alignment()
