#%%
from orth_scan import OrthScan
import numpy as np
import matplotlib.pyplot as plt
import h5py

with h5py.File('/home/zzhang/OneDrive/code/scanning-drift-corr/data_examples/nonlinear_drift_correction_synthetic_dataset_for_testing.mat', 'r') as f:
    img1 = f['image00deg'][:].T
    img2 = f['image90deg'][:].T
    img_gt = f['imageIdeal'][:].T


osc = OrthScan([img1, img2], scan_angles=[0, 90])
osc.run_linear_alignment()

# %%
