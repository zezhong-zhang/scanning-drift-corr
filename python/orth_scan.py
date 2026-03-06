import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_dilation
from scipy.signal import convolve, convolve2d
from tqdm import tqdm
from typing import List, Optional, Tuple
import warnings
# from python.utils import *
warnings.filterwarnings('ignore')

class OrthScan:
    def __init__(
        self,
        images,
        scan_angles=(0, 90),
        padding_scale=1.5,
        kde_sigma=0.5,
        edge_width=None,
        linear_search_steps=2,
    ):
        """
        Initialize the OrthScan class for scanning probe drift correction.
        
        Parameters:
        - images: List of 2D numpy arrays or 3D array
        - scan_angles: List of scan angles in degrees
        - padding_scale: Scale factor for output image padding
        - kde_sigma: Smoothing parameter for kernel density estimation
        - edge_width: Edge blending width (default: imageSize/128)
        - linear_search_steps: Number of linear drift search steps
        """
        self.scan_angles = np.array(scan_angles)
        self.padding_scale = padding_scale
        self.kde_sigma = kde_sigma
        
        # Convert to list if 3D array
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images[i] for i in range(images.shape[0])]
        
        # Store original image shape
        self.original_image_shape = images[0].shape
        
        # Calculate padded image size (multiple of 4)
        self.image_size = np.array([
            int(np.round(images[0].shape[0] * padding_scale / 4) * 4),
            int(np.round(images[0].shape[1] * padding_scale / 4) * 4)
        ])
        
        # Set edge width - stored as fraction matching MATLAB's sMerge.edgeWidth = 1/128
        if edge_width is None:
            self.edge_width = 1.0 / 128
        else:
            self.edge_width = edge_width
            
        self.num_images = len(scan_angles)
        
        # Initialize data arrays
        self.scan_lines = np.zeros((images[0].shape[0], images[0].shape[1], self.num_images))
        self.scan_origin = np.zeros((images[0].shape[0], 2, self.num_images))
        self.scan_direction = np.zeros((self.num_images, 2))
        self.image_transform = np.zeros((self.image_size[0], self.image_size[1], self.num_images))
        self.image_density = np.zeros((self.image_size[0], self.image_size[1], self.num_images))
        
        # Load images and calculate initial scan origins
        for i, img in enumerate(images):
            self.scan_lines[:, :, i] = img
            self._calculate_initial_scan_origins(i)
            
        # Setup for linear alignment
        self.time_index = np.linspace(-0.5, 0.5, self.scan_lines.shape[0])
        self.linear_search = np.linspace(-0.04, 0.04, 1 + 2 * linear_search_steps) * self.scan_lines.shape[0]
        self.linear_drift = [0.0, 0.0]
        self.ref_point = self.image_size // 2
        
        # Create Hanning window for correlation
        self.w2 = self._create_hanning_window()
        
        # Non-linear alignment attributes
        self.scan_active = None
        self.scan_origin_step = None
        self.stats = []
        
    def _calculate_initial_scan_origins(self, index):
        """Calculate initial scan origins and directions.
        Matches MATLAB: xy = [(1:height)' ones(height,1)]; 
        xy(:,1) = xy(:,1) - height/2; xy(:,2) = xy(:,2) - width/2;
        """
        height, width = self.scan_lines[:, :, index].shape
        
        # Match MATLAB: xy(:,1) = (1:height) - height/2
        #               xy(:,2) = ones(height,1) - width/2
        xy = np.column_stack([
            np.arange(1, height + 1) - height / 2,
            np.ones(height) - width / 2
        ])
        
        # Rotate by scan angle
        angle_rad = np.radians(self.scan_angles[index])
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        xy_rot = np.column_stack([
            xy[:, 0] * cos_a - xy[:, 1] * sin_a,
            xy[:, 0] * sin_a + xy[:, 1] * cos_a
        ])
        
        # Translate to padded image center: MATLAB uses imageSize/2
        xy_rot[:, 0] = xy_rot[:, 0] + self.image_size[0] / 2
        xy_rot[:, 1] = xy_rot[:, 1] + self.image_size[1] / 2
        
        # Remove fractional offset to align to pixel grid
        xy_rot[:, 0] = xy_rot[:, 0] - (xy_rot[0, 0] % 1)
        xy_rot[:, 1] = xy_rot[:, 1] - (xy_rot[0, 1] % 1)
        
        self.scan_origin[:, :, index] = xy_rot
        self.scan_direction[index] = [np.cos(angle_rad + np.pi/2), np.sin(angle_rad + np.pi/2)]
        
    def _create_hanning_window(self):
        """Create Hanning window for correlation."""
        N = self.scan_lines.shape[:2]
        hann_1d = lambda n: np.sin(np.pi * np.arange(1, n+1) / (n+1))**2
        window = np.outer(hann_1d(N[0]), hann_1d(N[1]))
        
        w2 = np.zeros(self.image_size)
        start_h = (self.image_size[0] - N[0]) // 2
        start_w = (self.image_size[1] - N[1]) // 2
        w2[start_h:start_h+N[0], start_w:start_w+N[1]] = window
        return w2
    
    def make_image(self, ind_image, ind_lines=None):
        """
        Generate resampled image from scan lines (equivalent to SPmakeImage).
        Matches MATLAB: t = 1:width, clamp to [1, imageSize-1].
        
        Parameters:
        - ind_image: Index of image to generate
        - ind_lines: Binary mask of which scan lines to use (default: all)
        """
        if ind_lines is None:
            ind_lines = np.ones(self.scan_lines.shape[0], dtype=bool)
            
        # Expand coordinates matching MATLAB: t = 1:width (1-based pixel steps)
        t = np.arange(1, self.scan_lines.shape[1] + 1).reshape(1, -1)
        x0 = self.scan_origin[ind_lines, 0, ind_image].reshape(-1, 1)
        y0 = self.scan_origin[ind_lines, 1, ind_image].reshape(-1, 1)
        
        x_ind = x0 + t * self.scan_direction[ind_image, 0]
        y_ind = y0 + t * self.scan_direction[ind_image, 1]
        
        # Clip to image boundaries matching MATLAB: max(min(xInd,imageSize-1),1)
        # Note: coordinates are in MATLAB-equivalent space [1, imageSize]
        # Valid bilinear range is [1, imageSize-1] so floor gives [1, imageSize-1]
        # and floor+1 gives [2, imageSize], both valid 1-based indices.
        # We use 0-based arrays, so subtract 1 for actual array indexing below.
        x_ind = np.clip(x_ind.flatten(), 1, self.image_size[0] - 1)
        y_ind = np.clip(y_ind.flatten(), 1, self.image_size[1] - 1)
        
        # Bilinear interpolation
        x_floor = np.floor(x_ind).astype(int)
        y_floor = np.floor(y_ind).astype(int)
        dx = x_ind - x_floor
        dy = y_ind - y_floor
        
        # Convert to 0-based indices for array access
        # MATLAB sub2ind uses 1-based, Python uses 0-based
        xF = x_floor - 1  # 0-based
        yF = y_floor - 1  # 0-based
        
        # Weights for bilinear interpolation
        weights = np.array([
            (1-dx)*(1-dy),
            dx*(1-dy),
            (1-dx)*dy,
            dx*dy
        ])
        
        # Indices for accumulation (0-based array indexing)
        indices = [
            np.ravel_multi_index((xF, yF), self.image_size),
            np.ravel_multi_index((xF+1, yF), self.image_size),
            np.ravel_multi_index((xF, yF+1), self.image_size),
            np.ravel_multi_index((xF+1, yF+1), self.image_size)
        ]
        
        # Generate image and density
        scan_data = self.scan_lines[ind_lines, :, ind_image].flatten()
        sig = np.zeros(np.prod(self.image_size))
        count = np.zeros(np.prod(self.image_size))
        
        for i, idx in enumerate(indices):
            np.add.at(sig, idx, weights[i] * scan_data)
            np.add.at(count, idx, weights[i])
            
        sig = sig.reshape(self.image_size)
        count = count.reshape(self.image_size)
        
        # Apply KDE smoothing matching MATLAB: fspecial('gaussian',2*r+1,sigma) + conv2
        if self.kde_sigma > 0:
            r = max(int(np.ceil(self.kde_sigma * 3)), 5)
            ax = np.arange(-r, r + 1)
            kernel_1d = np.exp(-ax**2 / (2 * self.kde_sigma**2))
            kernel_2d = np.outer(kernel_1d, kernel_1d)
            kernel_2d = kernel_2d / np.sum(kernel_2d)
            sig = convolve2d(sig, kernel_2d, mode='same', boundary='fill')
            count = convolve2d(count, kernel_2d, mode='same', boundary='fill')
            
        # Normalize
        mask = count > 0
        sig[mask] = sig[mask] / count[mask]
        self.image_transform[:, :, ind_image] = sig
        
        # Estimate density
        boundary = count == 0
        boundary[[0, -1], :] = True
        boundary[:, [0, -1]] = True
        dist = distance_transform_edt(~boundary)
        self.image_density[:, :, ind_image] = np.sin(np.minimum(dist/self.edge_width, 1) * np.pi/2)**2
        
    def run_linear_alignment(self):
        """Run complete linear alignment (SPmerge01)."""
        print("Running linear alignment...")
        self._search_linear_drift()
        self._refine_linear_drift()
        self._apply_linear_drift()
        self._initial_alignment()
        self.ref_point = self._find_reference_point()
        print(f"Linear alignment complete. Reference point: {self.ref_point}")
        
    def _search_linear_drift(self):
        """Initial linear drift search."""
        y_drift, x_drift = np.meshgrid(self.linear_search, self.linear_search)
        self.linear_search_score_1 = np.zeros_like(x_drift)
        
        for i in tqdm(range(len(self.linear_search)), desc="Linear drift search"):
            for j in range(len(self.linear_search)):
                # Apply drift
                xy_shift = np.column_stack([
                    self.time_index * x_drift[i, j],
                    self.time_index * y_drift[i, j]
                ])
                
                for k in range(2):
                    self.scan_origin[:, :, k] += xy_shift
                    self.make_image(k)
                    
                # Measure correlation
                fft1 = np.fft.fft2(self.w2 * self.image_transform[:, :, 0])
                fft2 = np.fft.fft2(self.w2 * self.image_transform[:, :, 1])
                m = fft1 * np.conj(fft2)
                corr = np.abs(np.fft.ifft2(np.sqrt(np.abs(m)) * np.exp(1j * np.angle(m))))
                self.linear_search_score_1[i, j] = np.max(corr)
                
                # Revert drift
                for k in range(2):
                    self.scan_origin[:, :, k] -= xy_shift
                    
    def _refine_linear_drift(self):
        """Refine linear drift estimate."""
        # Find best initial guess
        best_idx = np.unravel_index(np.argmax(self.linear_search_score_1), 
                                   self.linear_search_score_1.shape)
        
        # Refine around best guess
        step = self.linear_search[1] - self.linear_search[0]
        x_refine = self.linear_search[best_idx[0]] + np.linspace(-0.5, 0.5, len(self.linear_search)) * step
        y_refine = self.linear_search[best_idx[1]] + np.linspace(-0.5, 0.5, len(self.linear_search)) * step
        
        y_drift, x_drift = np.meshgrid(y_refine, x_refine)
        self.linear_search_score_2 = np.zeros_like(self.linear_search_score_1)
        
        for i in tqdm(range(len(x_refine)), desc="Refining linear drift"):
            for j in range(len(y_refine)):
                xy_shift = np.column_stack([
                    self.time_index * x_drift[i, j],
                    self.time_index * y_drift[i, j]
                ])
                
                for k in range(2):
                    self.scan_origin[:, :, k] += xy_shift
                    self.make_image(k)
                    
                fft1 = np.fft.fft2(self.w2 * self.image_transform[:, :, 0])
                fft2 = np.fft.fft2(self.w2 * self.image_transform[:, :, 1])
                m = fft1 * np.conj(fft2)
                corr = np.abs(np.fft.ifft2(np.sqrt(np.abs(m)) * np.exp(1j * np.angle(m))))
                self.linear_search_score_2[i, j] = np.max(corr)
                
                for k in range(2):
                    self.scan_origin[:, :, k] -= xy_shift
                    
        best_idx = np.unravel_index(np.argmax(self.linear_search_score_2), 
                                   self.linear_search_score_2.shape)
        self.linear_drift = [x_drift[best_idx], y_drift[best_idx]]
        
    def _apply_linear_drift(self):
        """Apply linear drift to all images."""
        xy_shift = np.column_stack([
            self.time_index * self.linear_drift[0],
            self.time_index * self.linear_drift[1]
        ])
        
        for i in range(self.num_images):
            self.scan_origin[:, :, i] += xy_shift
            self.make_image(i)
            
    def _initial_alignment(self):
        """Initial alignment using phase correlation."""
        dxy = np.zeros((self.num_images, 2))
        G1 = np.fft.fft2(self.w2 * self.image_transform[:, :, 0])
        
        for i in range(1, self.num_images):
            G2 = np.fft.fft2(self.w2 * self.image_transform[:, :, i])
            m = G1 * np.conj(G2)
            corr = np.abs(np.fft.ifft2(np.sqrt(np.abs(m)) * np.exp(1j * np.angle(m))))
            
            peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
            # Convert peak index to shift using 0-based indexing
            # Phase correlation gives shift to align image i to image i-1, so we flip the sign
            dx = ((peak_idx[0] + self.image_size[0]//2) % self.image_size[0] - self.image_size[0]//2)
            dy = ((peak_idx[1] + self.image_size[1]//2) % self.image_size[1] - self.image_size[1]//2)
            
            dxy[i] = dxy[i-1] + [dx, dy]
            G1 = G2
            
        # Center the shifts
        dxy -= np.mean(dxy, axis=0)
        
        # Apply shifts to scan origins
        for i in range(self.num_images):
            self.scan_origin[:, 0, i] += dxy[i, 0]
            self.scan_origin[:, 1, i] += dxy[i, 1]
            self.make_image(i)
            
    def _find_reference_point(self):
        """Find optimal reference point."""
        return np.round(self.image_size / 2).astype(int)
    
    def refine_scan_origins(self, max_iterations=32, initial_steps=4, 
                           density_cutoff=0.8, initial_smoothing=0, 
                           refinement_smoothing=8, global_shift=True,
                           point_ordering=True):
        """
        Non-linear refinement of scan origins (SPmerge02).
        
        Parameters:
        - max_iterations: Maximum refinement iterations
        - initial_steps: Number of initial alignment steps
        - density_cutoff: Density threshold for valid regions
        - initial_smoothing: Smoothing window for initial alignment
        - refinement_smoothing: Smoothing window for main refinement
        - global_shift: Enable global phase correlation
        - point_ordering: Enforce scanline ordering
        """
        print(f"Starting non-linear refinement (up to {max_iterations} iterations)...")
        
        # Setup
        self.stats = np.zeros((max_iterations + 1, 2))
        dist_start = np.mean(self.scan_lines.shape[:2]) / 16
        initial_shift_max = 0.25
        refine_step_size = 0.5
        step_reduce = 0.5
        pixels_threshold = 0.1
        min_global_shift = 4
        
        # Initialize scan_origin_step
        self.scan_origin_step = np.ones((self.scan_origin.shape[0], self.num_images)) * refine_step_size
        
        # Setup smoothing kernel for initial alignment
        if initial_smoothing > 0:
            self._setup_smoothing(initial_smoothing)
        
        # Initial alignment
        if self.scan_active is None or initial_steps > 0:
            self._initial_nonlinear_alignment(initial_steps, dist_start, 
                                             initial_shift_max, density_cutoff,
                                             initial_smoothing)
        
        # Setup smoothing for main refinement
        if refinement_smoothing > 0:
            self._setup_smoothing(refinement_smoothing)
        
        # Main refinement loop
        self._main_refinement_loop(max_iterations, density_cutoff, global_shift,
                                  point_ordering, step_reduce, pixels_threshold,
                                  min_global_shift, refinement_smoothing)
        
        # Final image generation
        for i in range(self.num_images):
            self.make_image(i)
            
        print(f"Non-linear refinement complete. Final mean abs diff: {self.stats[-1, 1]:.4f}")
        
    def _setup_smoothing(self, window_size):
        """Setup kernel for origin smoothing."""
        if window_size > 0:
            r = int(np.ceil(3 * window_size))
            v = np.arange(-r, r + 1)
            self.kde_origin = np.exp(-v**2 / (2 * window_size**2))
            self.kde_origin = self.kde_origin.reshape(-1, 1, 1)
        else:
            self.kde_origin = np.array([[[1.0]]])
            
        self.basis_or = np.column_stack([
            np.ones(self.scan_lines.shape[0]),
            np.arange(1, self.scan_lines.shape[0] + 1)  # MATLAB uses 1:N
        ])
        
    def _initial_nonlinear_alignment(self, initial_steps, dist_start, 
                                    initial_shift_max, density_cutoff,
                                    smoothing_window):
        """Perform initial non-linear alignment."""
        
        for step in tqdm(range(initial_steps), desc="Initial non-linear alignment"):
            self.scan_active = np.zeros((self.scan_lines.shape[0], self.num_images), dtype=bool)
            ind_start = np.zeros(self.num_images, dtype=int)
            
            # Find starting scanlines
            for i in range(self.num_images):
                v = np.array([-self.scan_direction[i, 1], self.scan_direction[i, 0]])
                origins = self.scan_origin[:, :, i]
                c = -np.dot(self.ref_point, v)
                dist = np.abs(v[0] * origins[:, 0] + v[1] * origins[:, 1] + c) / np.linalg.norm(v)
                ind_start[i] = np.argmin(dist)
                self.scan_active[dist < dist_start, i] = True
                
            # Align each image
            for i in range(self.num_images):
                # Find orthogonal image to align to
                dot_products = np.abs(np.sum(self.scan_direction * self.scan_direction[i], axis=1))
                ind_align = np.argmin(dot_products)
                
                # Generate alignment image
                self.make_image(ind_align, self.scan_active[:, ind_align])
                image_align = self.image_transform[:, :, ind_align] * (self.image_density[:, :, ind_align] > density_cutoff)
                
                # Align origins line by line
                xy_step = np.mean(np.diff(self.scan_origin[:, :, i], axis=0), axis=0)
                ind_aligned = np.zeros(self.scan_lines.shape[0], dtype=bool)
                ind_aligned[ind_start[i]] = True
                
                while not np.all(ind_aligned):
                    # Get next scanlines to align
                    v = binary_dilation(ind_aligned.reshape(-1, 1)).flatten()
                    v[ind_aligned] = False
                    ind_move = np.where(v)[0]
                    inds_active = np.where(ind_aligned)[0]
                    
                    for idx in ind_move:
                        # Find nearest aligned scanline
                        nearest = inds_active[np.argmin(np.abs(idx - inds_active))]
                        xy_or = self.scan_origin[nearest, :, i] + xy_step * (idx - nearest)
                        
                        # Test different positions (5-point plus shape)
                        best_score = np.inf
                        best_shift = np.array([0, 0])
                        
                        test_shifts = np.array([
                            [0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]
                        ])
                        
                        for shift in test_shifts:
                            dx, dy = shift
                            test_or = xy_or + np.array([dx, dy]) * initial_shift_max
                            score = self._calc_scanline_score(test_or, i, idx, image_align)
                            if score < best_score:
                                best_score = score
                                best_shift = np.array([dx, dy]) * initial_shift_max
                                    
                        self.scan_origin[idx, :, i] = xy_or + best_shift
                        ind_aligned[idx] = True
                        
            # Apply smoothing if needed
            if smoothing_window > 0:
                self._smooth_origins()
                
    def _main_refinement_loop(self, max_iterations, density_cutoff, global_shift,
                             point_ordering, step_reduce, pixels_threshold,
                             min_global_shift, smoothing_window):
        """Main refinement iterations."""
        
        align_step = 0
        while align_step < max_iterations:
            pixels_moved = 0
            
            # Generate all images with current alignment
            for i in range(self.num_images):
                self.make_image(i)
            
            # Calculate mean absolute difference before refinement
            mean_img = np.mean(self.image_transform, axis=2)
            diff = np.mean(np.abs(self.image_transform - mean_img[:, :, np.newaxis]), axis=2)
            mask = np.min(self.image_density, axis=2) > density_cutoff
            if np.any(mask):
                mean_abs_diff = np.mean(diff[mask]) / (np.mean(np.abs(self.scan_lines)) + 1e-10)
            else:
                mean_abs_diff = 1.0
            self.stats[align_step] = [align_step, mean_abs_diff]
            
            # Apply global phase correlation if enabled (CRITICAL for alignment!)
            if global_shift:
                print(f"\nIteration {align_step + 1}: Checking global alignment...")
                
                # Save current state in case we need to revert
                scan_origin_backup = self.scan_origin.copy()
                scan_origin_step_backup = self.scan_origin_step.copy()
                mean_abs_diff_current = mean_abs_diff
                
                global_pixels = self._global_phase_correlation(density_cutoff, min_global_shift)
                
                if global_pixels > 0:
                    # Recalculate mean abs diff after global shift
                    for i in range(self.num_images):
                        self.make_image(i)
                    mean_img = np.mean(self.image_transform, axis=2)
                    diff = np.mean(np.abs(self.image_transform - mean_img[:, :, np.newaxis]), axis=2)
                    if np.any(mask):
                        mean_abs_diff_new = np.mean(diff[mask]) / (np.mean(np.abs(self.scan_lines)) + 1e-10)
                        
                        # Check if global shift improved the result
                        if mean_abs_diff_new < mean_abs_diff_current:
                            self.stats[align_step, 1] = mean_abs_diff_new
                            pixels_moved += global_pixels
                            print(f"  Global shift accepted. MAD improved: {mean_abs_diff_current:.4f} -> {mean_abs_diff_new:.4f}")
                        else:
                            # Revert if not improved
                            self.scan_origin = scan_origin_backup
                            self.scan_origin_step = scan_origin_step_backup
                            # Restore images
                            for i in range(self.num_images):
                                self.make_image(i)
                            print(f"  Global shift reverted. MAD worsened: {mean_abs_diff_current:.4f} -> {mean_abs_diff_new:.4f}")
                    else:
                         self.scan_origin = scan_origin_backup
                         self.scan_origin_step = scan_origin_step_backup
                         for i in range(self.num_images):
                                self.make_image(i)
            
            # Refine individual scanlines for each image
            print(f"Iteration {align_step + 1}: Refining scanlines (MAD={mean_abs_diff*100:.2f}%)...")
            
            for i in range(self.num_images):
                # Generate alignment target (mean of other images)
                inds_align = list(range(self.num_images))
                inds_align.remove(i)
                
                if len(inds_align) > 0:
                    # Create alignment target from other images
                    image_align = np.zeros(self.image_size)
                    weight_total = np.zeros(self.image_size)
                    
                    for j in inds_align:
                        weight = self.image_density[:, :, j] > density_cutoff
                        image_align += self.image_transform[:, :, j] * weight
                        weight_total += weight
                    
                    # Normalize
                    mask = weight_total > 0
                    image_align[mask] = image_align[mask] / weight_total[mask]
                    image_align[~mask] = np.mean(image_align[mask]) if np.any(mask) else 0
                else:
                    # If only one image, skip refinement
                    continue
                
                # Setup for point ordering if needed
                if point_ordering:
                    n = np.array([self.scan_direction[i, 1], -self.scan_direction[i, 0]])
                    v_param = n[0] * self.scan_origin[:, 0, i] + n[1] * self.scan_origin[:, 1, i]
                
                # Refine each scanline
                for j in range(self.scan_lines.shape[0]):
                    best_score = np.inf
                    best_or = self.scan_origin[j, :, i].copy()
                    original_or = self.scan_origin[j, :, i].copy()
                    best_idx = 0  # Default: center (no move)
                    
                    # Test 5 positions (center and 4 cardinal directions)
                    # Matches MATLAB: orTest = repmat(scanOr(a1,1:2,a0),[5 1]) + dxy * scanOrStep(a1,a0)
                    dxy = np.array([
                        [0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]
                    ])
                    or_test = np.tile(self.scan_origin[j, :, i], (5, 1)) + dxy * self.scan_origin_step[j, i]
                    
                    # Apply ordering constraint to ALL test positions before scoring
                    # Matches MATLAB: compute bounds, then clamp all positions
                    if point_ordering:
                        v_test = n[0] * or_test[:, 0] + n[1] * or_test[:, 1]
                        if j == 0:
                            v_bound = [-np.inf, v_param[j+1]]
                        elif j == self.scan_lines.shape[0] - 1:
                            v_bound = [v_param[j-1], np.inf]
                        else:
                            v_bound = [v_param[j-1], v_param[j+1]]
                        for a2 in range(5):
                            if v_test[a2] < v_bound[0]:
                                or_test[a2, :] += n * (v_bound[0] - v_test[a2])
                            elif v_test[a2] > v_bound[1]:
                                or_test[a2, :] += n * (v_bound[1] - v_test[a2])
                    
                    # Score all 5 test positions
                    for a2 in range(5):
                        score = self._calc_scanline_score(or_test[a2], i, j, image_align)
                        if score < best_score:
                            best_score = score
                            best_or = or_test[a2].copy()
                            best_idx = a2
                    
                    # Apply best position: MATLAB uses scanOr(a1,:,a0) = orTest(ind,:)
                    if best_idx == 0:
                        # No improvement (center position won), reduce step size
                        self.scan_origin_step[j, i] *= step_reduce
                    else:
                        # Apply the constraint-adjusted position directly
                        pixels_moved += np.linalg.norm(best_or - original_or)
                        self.scan_origin[j, :, i] = best_or
            
            # Apply smoothing if specified
            if smoothing_window > 0:
                self._smooth_origins()
            
            print(f"  Pixels moved: {pixels_moved:.2f}")
            
            # Check convergence
            if pixels_moved / self.num_images < pixels_threshold:
                print(f"Converged after {align_step + 1} iterations")
                break
            
            align_step += 1
        
        # Store final statistics
        self.stats = self.stats[:align_step+1]  # Trim unused entries
        
    def _calc_scanline_score(self, origin, img_idx, line_idx, image_align):
        """Calculate alignment score for a scanline.
        Matches MATLAB: inds = 1:width, clamp to [1, imageSize-1], bilinear interpolation.
        """
        inds = np.arange(1, self.scan_lines.shape[1] + 1)
        x_ind = origin[0] + inds * self.scan_direction[img_idx, 0]
        y_ind = origin[1] + inds * self.scan_direction[img_idx, 1]
        
        # Clip to boundaries matching MATLAB: max(min(xInd,imageSize-1),1)
        x_ind = np.clip(x_ind, 1, self.image_size[0] - 1)
        y_ind = np.clip(y_ind, 1, self.image_size[1] - 1)
        
        # Bilinear interpolation
        x_floor = np.floor(x_ind).astype(int)
        y_floor = np.floor(y_ind).astype(int)
        dx = x_ind - x_floor
        dy = y_ind - y_floor
        
        # Convert to 0-based indices for array access
        xF = x_floor - 1
        yF = y_floor - 1
        
        # Sample from alignment image using bilinear interpolation
        # Matches MATLAB calcScore: sub2ind based bilinear
        sample = (image_align[xF, yF] * (1-dx) * (1-dy) +
                 image_align[xF+1, yF] * dx * (1-dy) +
                 image_align[xF, yF+1] * (1-dx) * dy +
                 image_align[xF+1, yF+1] * dx * dy)
        
        return np.sum(np.abs(sample - self.scan_lines[line_idx, :, img_idx]))
    
    def _global_phase_correlation(self, density_cutoff, min_shift):
        """Apply global phase correlation alignment.
        Matches MATLAB SPmerge02: uses densityDist for image 0, hardcoded 64 for others.
        """
        pixels_moved = 0
        intensity_median = np.median(self.scan_lines)
        
        # Reference image — use densityDist = mean(size)/32
        density_dist = np.mean(self.scan_lines.shape[:2]) / 32
        density_mask = distance_transform_edt(self.image_density[:, :, 0] < density_cutoff)
        density_mask = np.sin(np.minimum(density_mask / density_dist, 1) * np.pi/2)**2
        
        ref_img = self.image_transform[:, :, 0] * density_mask + (1 - density_mask) * intensity_median
        ref_fft = np.fft.fft2(ref_img)
        
        for i in range(1, self.num_images):
            # Current image — MATLAB uses hardcoded 64 for non-reference images
            density_mask = distance_transform_edt(self.image_density[:, :, i] < density_cutoff)
            density_mask = np.sin(np.minimum(density_mask / 64.0, 1) * np.pi/2)**2
            
            curr_img = self.image_transform[:, :, i] * density_mask + (1 - density_mask) * intensity_median
            curr_fft = np.conj(np.fft.fft2(curr_img))
            
            # Phase correlation
            phase_corr = np.abs(np.fft.ifft2(np.exp(1j * np.angle(ref_fft * curr_fft))))
            
            # Find peak — matches MATLAB's mod(xInd-1+imageSize/2, imageSize) - imageSize/2
            peak_idx = np.unravel_index(np.argmax(phase_corr), phase_corr.shape)
            dx = ((peak_idx[0] + self.image_size[0]//2) % self.image_size[0] - self.image_size[0]//2)
            dy = ((peak_idx[1] + self.image_size[1]//2) % self.image_size[1] - self.image_size[1]//2)
            
            # Apply shift if significant
            # Match MATLAB boundary check: min(xNew) >= 1 && max(xNew) < imageSize-1
            if abs(dx) + abs(dy) > min_shift:
                x_new = self.scan_origin[:, 0, i] + dx
                y_new = self.scan_origin[:, 1, i] + dy
                
                if (np.min(x_new) >= 1 and np.max(x_new) < self.image_size[0] - 1 and
                    np.min(y_new) >= 1 and np.max(y_new) < self.image_size[1] - 1):
                    self.scan_origin[:, 0, i] = x_new
                    self.scan_origin[:, 1, i] = y_new
                    self.scan_origin_step[:, i] = 0.5  # Reset step size
                    pixels_moved += np.sqrt(dx**2 + dy**2) * self.scan_lines.shape[0]
                    
        return pixels_moved
    
    def _smooth_origins(self):
        """Apply smoothing to scan origins."""
        scan_or_linear = np.zeros_like(self.scan_origin)
        
        for i in range(self.num_images):
            # Fit linear trend
            ppx = np.linalg.lstsq(self.basis_or, self.scan_origin[:, 0, i], rcond=None)[0]
            ppy = np.linalg.lstsq(self.basis_or, self.scan_origin[:, 1, i], rcond=None)[0]
            scan_or_linear[:, 0, i] = self.basis_or @ ppx
            scan_or_linear[:, 1, i] = self.basis_or @ ppy
            
        # Remove linear trend
        self.scan_origin -= scan_or_linear
        
        # Apply smoothing
        if len(self.kde_origin) > 1:
            kde_norm = 1.0 / convolve(np.ones((self.scan_origin.shape[0], 1, 1)), 
                                     self.kde_origin, mode='same')
            self.scan_origin = convolve(self.scan_origin, self.kde_origin, mode='same') * kde_norm
            
        # Restore linear trend
        self.scan_origin += scan_or_linear
    
    def generate_final_image(self, upsample_factor=2, fourier_weighting=True, 
                           downsample_output=True, boundary_width=8, return_intermediate=False, debug=False):
        """
        Generate final fused image (SPmerge03).
        
        Parameters:
        - upsample_factor: Upsampling factor for KDE
        - fourier_weighting: Use Fourier domain weighting
        - downsample_output: Downsample to original resolution
        - boundary_width: Edge blending width in pixels
        
        Returns:
        - final_image: Drift-corrected fused image
        """
        print("Generating final fused image...")
        
        # Initialize arrays
        upsampled_size = tuple(np.array(self.image_size) * upsample_factor)
        signal_array = np.zeros((*upsampled_size, self.num_images))
        density_array = np.zeros_like(signal_array)
        
        # Generate KDE kernel matching MATLAB's approach:
        # x = makeFourierCoords(N, 1/imageSize) gives pixel positions
        # kernel = fft2(exp(-x^2/(2*sigma^2)) * exp(-y^2/(2*sigma^2)))
        # This constructs a spatial Gaussian on the grid and FFTs it.
        def make_fourier_coords(N, pSize):
            """Match MATLAB's makeFourierCoords: pixel-space coordinates in FFT order."""
            if N % 2 == 0:
                q = np.roll(np.arange(-N//2, N//2) / (N * pSize), -N//2)
            else:
                q = np.roll((np.arange(-N//2+0.5, N//2+0.5)) / ((N-1) * pSize), int(-N//2+0.5))
            return q
        
        x = make_fourier_coords(upsampled_size[0], 1.0/self.image_size[0]).reshape(-1, 1)
        y = make_fourier_coords(upsampled_size[1], 1.0/self.image_size[1]).reshape(1, -1)
        
        kde_sigma = self.kde_sigma
        kernel_spatial = np.exp(-x**2 / (2 * kde_sigma**2)) * np.exp(-y**2 / (2 * kde_sigma**2))
        kernel_fft = np.fft.fft2(kernel_spatial)
        
        # Density smoothing kernel (matching MATLAB)
        sigma_density = 4
        density_spatial = (np.exp(-x**2 / (2 * sigma_density**2)) * 
                          np.exp(-y**2 / (2 * sigma_density**2)) /
                          (2 * np.pi * sigma_density**2 * upsample_factor**2))
        density_smooth_fft = np.fft.fft2(density_spatial)
        
        # Process each image
        for i in tqdm(range(self.num_images), desc="Processing images for fusion"):
            # Resample at higher resolution
            # Match MATLAB: t = 1:width, xInd = x0*upsample - (upsample-1)/2 + t*scanDir*upsample
            t = np.arange(1, self.scan_lines.shape[1] + 1)
            x0 = self.scan_origin[:, 0, i].reshape(-1, 1)
            y0 = self.scan_origin[:, 1, i].reshape(-1, 1)
            
            x_ind = x0 * upsample_factor - (upsample_factor - 1) / 2 + t * self.scan_direction[i, 0] * upsample_factor
            y_ind = y0 * upsample_factor - (upsample_factor - 1) / 2 + t * self.scan_direction[i, 1] * upsample_factor
            
            # Clip to boundaries matching MATLAB: min(max(xInd,1),imageSize*upsample-1)
            x_ind = np.clip(x_ind.flatten(), 1, upsampled_size[0] - 1)
            y_ind = np.clip(y_ind.flatten(), 1, upsampled_size[1] - 1)
            
            # Bilinear interpolation
            x_floor = np.floor(x_ind).astype(int)
            y_floor = np.floor(y_ind).astype(int)
            dx = x_ind - x_floor
            dy = y_ind - y_floor
            
            # Convert to 0-based indices for array access
            xF = x_floor - 1
            yF = y_floor - 1
            
            if debug:
                print("x_ind:", x_ind[:10])
                print("y_ind:", y_ind[:10])
                print("dx:", dx[:10])
                print("dy:", dy[:10])
            
            # Accumulate signal and density
            scan_data = self.scan_lines[:, :, i].flatten()
            sig = np.zeros(np.prod(upsampled_size))
            dens = np.zeros(np.prod(upsampled_size))
            
            weights = [(1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy]
            indices = [
                np.ravel_multi_index((xF, yF), upsampled_size),
                np.ravel_multi_index((xF + 1, yF), upsampled_size),
                np.ravel_multi_index((xF, yF + 1), upsampled_size),
                np.ravel_multi_index((xF + 1, yF + 1), upsampled_size)
            ]
            
            for idx, w in zip(indices, weights):
                np.add.at(sig, idx, scan_data * w)
                np.add.at(dens, idx, w)
                
            sig = sig.reshape(upsampled_size)
            dens = dens.reshape(upsampled_size)
            
            # Apply KDE
            sig_fft = np.fft.fft2(sig)
            dens_fft = np.fft.fft2(dens)
            signal_array[:, :, i] = np.real(np.fft.ifft2(sig_fft * kernel_fft))
            density_array[:, :, i] = np.real(np.fft.ifft2(dens_fft * kernel_fft))
            
        # Normalize by density
        mask = density_array > 1e-8
        signal_array[mask] = signal_array[mask] / density_array[mask]
        
        # Apply smooth density estimation
        intensity_median = np.median(self.scan_lines)
        for i in range(self.num_images):
            dens_smooth = np.real(np.fft.ifft2(np.fft.fft2(np.minimum(density_array[:, :, i], 2)) * density_smooth_fft))
            dist = distance_transform_edt(dens_smooth < 0.5)
            density_array[:, :, i] = np.sin(np.minimum(dist / (boundary_width * upsample_factor), 1) * np.pi/2)**2
            signal_array[:, :, i] = signal_array[:, :, i] * density_array[:, :, i] + \
                                   (1 - density_array[:, :, i]) * intensity_median
        
        # Combine images
        if fourier_weighting:
            # Fourier weighting based on scan direction
            # Match MATLAB: qx = makeFourierCoords(size,1), qy = makeFourierCoords(size,1)
            qx_w = make_fourier_coords(upsampled_size[0], 1.0)
            qy_w = make_fourier_coords(upsampled_size[1], 1.0)
            qya_w, qxa_w = np.meshgrid(qy_w, qx_w)
            q_theta = np.arctan2(qya_w, qxa_w)
            final_fft = np.zeros(upsampled_size, dtype=complex)
            weight_total = np.zeros(upsampled_size)
            
            for i in range(self.num_images):
                theta_scan = np.arctan2(self.scan_direction[i, 1], self.scan_direction[i, 0])
                q_weight = np.cos(q_theta - theta_scan)**2
                q_weight[0, 0] = 1  # DC component
                
                img_fft = np.fft.fft2(signal_array[:, :, i])
                final_fft += img_fft * q_weight
                weight_total += q_weight
                
            final_image = np.real(np.fft.ifft2(final_fft / (weight_total + 1e-10)))
        else:
            # Simple density-weighted average
            density_prod = np.prod(density_array, axis=2)
            final_image = np.mean(signal_array, axis=2) * density_prod + \
                         (1 - density_prod) * intensity_median
        
        # Apply global density mask
        global_density = np.prod(density_array, axis=2)
        final_image = final_image * global_density + (1 - global_density) * intensity_median
        
        # Downsample if requested
        if downsample_output and upsample_factor > 1:
            # Fourier cropping for downsampling
            # Match MATLAB: xVec = [(1:N/2) ((-N/2+1):0) + N*upsample]
            # In 0-based: [0:N/2-1] and [-N/2+1:0] mapped to upsampled indices
            x_vec = list(range(self.image_size[0]//2)) + \
                   list(range(-self.image_size[0]//2 + 1, 1))  # includes 0
            y_vec = list(range(self.image_size[1]//2)) + \
                   list(range(-self.image_size[1]//2 + 1, 1))  # includes 0
            
            x_vec = [(x + upsampled_size[0]) % upsampled_size[0] for x in x_vec]
            y_vec = [(y + upsampled_size[1]) % upsampled_size[1] for y in y_vec]
            
            final_fft = np.fft.fft2(final_image)
            final_image = np.real(np.fft.ifft2(final_fft[np.ix_(x_vec, y_vec)])) / upsample_factor**2
            
        print("Image fusion complete!")
        if return_intermediate:
            return final_image, signal_array, density_array
        else:
            return final_image
    
    def diagnostic_plot(self, stage="current"):
        """
        Create diagnostic plots to visualize alignment issues.
        
        Parameters:
        - stage: Stage of processing ("initial", "linear", "nonlinear", "final")
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Individual transformed images
        for i in range(min(self.num_images, 2)):
            ax = axes[0, i]
            img = self.image_transform[:, :, i]
            density = self.image_density[:, :, i]
            masked_img = img * (density > 0.5)
            ax.imshow(masked_img, cmap='gray')
            ax.set_title(f'Image {i+1} ({self.scan_angles[i]}°)')
            ax.axis('off')
        
        # Overlay of both images
        ax = axes[0, 2]
        if self.num_images >= 2:
            img1 = self.image_transform[:, :, 0] * (self.image_density[:, :, 0] > 0.5)
            img2 = self.image_transform[:, :, 1] * (self.image_density[:, :, 1] > 0.5)
            
            # Normalize for visualization
            img1_norm = (img1 - np.mean(img1[img1 > 0])) / (np.std(img1[img1 > 0]) + 1e-10)
            img2_norm = (img2 - np.mean(img2[img2 > 0])) / (np.std(img2[img2 > 0]) + 1e-10)
            
            # Create RGB overlay (img1=red, img2=green, overlap=yellow)
            rgb_overlay = np.zeros((*img1.shape, 3))
            rgb_overlay[:, :, 0] = np.clip(img1_norm + 2, 0, 1)  # Red channel
            rgb_overlay[:, :, 1] = np.clip(img2_norm + 2, 0, 1)  # Green channel
            
            ax.imshow(rgb_overlay)
            ax.set_title('Overlay (Red=Img1, Green=Img2, Yellow=Overlap)')
        ax.axis('off')
        
        # Difference image
        ax = axes[1, 0]
        if self.num_images >= 2:
            diff = np.abs(img1_norm - img2_norm)
            im = ax.imshow(diff, cmap='hot', vmin=0, vmax=2)
            ax.set_title('Absolute Difference')
            plt.colorbar(im, ax=ax, fraction=0.046)
        ax.axis('off')
        
        # Mean image
        ax = axes[1, 1]
        mean_img = np.mean(self.image_transform, axis=2)
        density_prod = np.prod(self.image_density, axis=2)
        mean_img = mean_img * (density_prod > 0.5)
        ax.imshow(mean_img, cmap='gray')
        ax.set_title('Mean Image')
        ax.axis('off')
        
        # Scan origins
        ax = axes[1, 2]
        ax.set_aspect('equal')
        colors = ['red', 'blue', 'green', 'orange']
        for i in range(self.num_images):
            ax.scatter(self.scan_origin[:, 1, i], self.scan_origin[:, 0, i],
                      c=colors[i % len(colors)], s=2, alpha=0.7,
                      label=f'Image {i+1}')
        ax.scatter(self.ref_point[1], self.ref_point[0], 
                  c='yellow', s=200, marker='+', linewidth=3,
                  label='Reference')
        ax.set_xlim([0, self.image_size[1]])
        ax.set_ylim([self.image_size[0], 0])  # Invert y-axis for image coordinates
        ax.set_title('Scan Origins')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Diagnostic Plot - Stage: {stage}', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # Print alignment metrics
        print(f"\n--- Alignment Metrics ({stage}) ---")
        if self.num_images >= 2:
            # Calculate centroid shift between images
            centroid1 = np.mean(self.scan_origin[:, :, 0], axis=0)
            centroid2 = np.mean(self.scan_origin[:, :, 1], axis=0)
            shift = centroid2 - centroid1
            print(f"Centroid shift between images: dx={shift[0]:.2f}, dy={shift[1]:.2f} pixels")
            
            # Calculate mean absolute difference
            mean_img = np.mean(self.image_transform, axis=2)
            diff_img = np.mean(np.abs(self.image_transform - mean_img[:, :, np.newaxis]), axis=2)
            mask = np.min(self.image_density, axis=2) > 0.5
            if np.any(mask):
                mad = np.mean(diff_img[mask]) / (np.mean(np.abs(self.scan_lines)) + 1e-10)
                print(f"Mean absolute difference: {mad*100:.2f}%")
            
            # Check for phase correlation peak
            G1 = np.fft.fft2(self.w2 * self.image_transform[:, :, 0])
            G2 = np.fft.fft2(self.w2 * self.image_transform[:, :, 1])
            m = G1 * np.conj(G2)
            corr = np.abs(np.fft.ifft2(np.sqrt(np.abs(m)) * np.exp(1j * np.angle(m))))
            peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
            print(f"Phase correlation peak at: ({peak_idx[0]}, {peak_idx[1]})")
            print(f"Expected peak for perfect alignment: ({0}, {0}) or near image center")
        
        print("-" * 40)

    def visualize_alignment(self, final_image=None, save_path=None):
        """
        Visualize alignment results similar to paper figures.
        Overlays scanline origins on the fused image.
        
        Parameters:
        - final_image: Optional fused image (if None, uses mean of transformed images)
        - save_path: Optional path to save the figure
        """
        if final_image is None:
            final_image = np.mean(self.image_transform, axis=2)
            
        plt.figure(figsize=(10, 10))
        
        # Plot fused image
        plt.imshow(final_image, cmap='gray', origin='upper')
        
        # Plot scanline origins
        colors = ['red', 'green', 'blue', 'orange']
        
        # For each image, plot its scanline origins
        for i in range(self.num_images):
            # scan_origin is (num_lines, 2, num_images)
            # origin[0] is row (y), origin[1] is col (x)
            # Matplotlib scatter takes (x, y)
            
            # Subsample lines for clarity if there are too many
            step = max(1, self.scan_lines.shape[0] // 100)
            
            # Get valid origins
            valid_origins = self.scan_origin[::step, :, i]
            
            plt.scatter(valid_origins[:, 1], valid_origins[:, 0], 
                       c=colors[i % len(colors)], 
                       s=10, 
                       label=f'Scan {i+1} Origins',
                       alpha=0.6,
                       edgecolor='none')
                       
            # Draw lines connecting origins to visualize drift path
            plt.plot(valid_origins[:, 1], valid_origins[:, 0],
                    c=colors[i % len(colors)],
                    alpha=0.3,
                    linewidth=1)

        plt.title('Drift Correction Visualization (Top View)')
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.legend()
        # plt.axis('equal') # Commented out to potentially avoid layout issues, image aspect is usually square
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()


# Example usage function
def run_spmerge(images, scan_angles=(0, 90), max_iterations=32):
    """
    Complete SPmerge pipeline.
    
    Parameters:
    - images: List of 2D arrays or 3D array of images
    - scan_angles: List of scan angles in degrees
    - max_iterations: Maximum refinement iterations
    
    Returns:
    - scanner: OrthScan object with results
    - final_image: Drift-corrected fused image
    """
    # Initialize
    scanner = OrthScan(images, scan_angles=scan_angles)
    
    # Step 1: Linear alignment (SPmerge01)
    scanner.run_linear_alignment()
    
    # Step 2: Non-linear refinement (SPmerge02)
    scanner.refine_scan_origins(max_iterations=max_iterations, 
                               initial_steps=4,
                               refinement_smoothing=8)
    
    # Step 3: Generate final image (SPmerge03)
    final_image = scanner.generate_final_image(upsample_factor=2, 
                                              fourier_weighting=True)
    
    # Display results
    scanner.plot_results(final_image)
    
    return scanner, final_image
