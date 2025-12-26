#%%
from orth_scan import OrthScan
import numpy as np
import matplotlib.pyplot as plt
import h5py

with h5py.File('/Users/zhangzz/code/scanning-drift-corr/data_examples/nonlinear_drift_correction_synthetic_dataset_for_testing.mat', 'r') as f:
    img1 = f['image00deg'][:].T
    img2 = f['image90deg'][:].T
    img_gt = f['imageIdeal'][:].T


osc = OrthScan([img1, img2], scan_angles=[0, 90])
osc.run_linear_alignment()

# %%


# Step 2: Non-linear refinement (SPmerge02)
osc.refine_scan_origins(max_iterations=32, 
                        initial_steps=4,
                        refinement_smoothing=8)

# Step 3: Generate final image (SPmerge03)
final_image = osc.generate_final_image(upsample_factor=2, 
                                        fourier_weighting=True)
# Display results
# %%
osc.diagnostic_plot()# %%


# %%
