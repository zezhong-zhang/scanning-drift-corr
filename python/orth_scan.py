import numpy as np
import matplotlib.pyplot as plt
from utils import fft2d, ifft2d
from scipy.ndimage import gaussian_filter, distance_transform_edt
from tqdm import tqdm


class OrthScan:
    def __init__(
        self,
        images,
        scan_angles=[0, 90],
        low_pass_sigma=32,
        report_progress=True,
        padding_scale=1.5,
        kde_sigma=0.5,
        edge_width=1 / 128,
        linear_search_steps=2,
    ):
        """
        Initialize the DriftCorrector class to align and drift correct scanning probe images.

        Parameters:
        - images: 3D numpy array or a list of 2D numpy arrays (the input images).
        - scan_angles: list of angles in degrees.
        - low_pass_sigma: sigma value for low-pass filter (default is 32).
        - report_progress: boolean to show progress in the console.
        - padding_scale: scale factor for padding the images.
        - kde_sigma: smoothing factor for KDE.
        - edge_width: width of the blending edge relative to input images.
        - linear_search_steps: number of steps for the linear drift search.
        """
        self.scan_angles = scan_angles
        self.low_pass_sigma = low_pass_sigma
        self.report_progress = report_progress
        self.padding_scale = padding_scale
        self.kde_sigma = kde_sigma
        self.edge_width = edge_width

        # Initialize the sMerge struct
        self.image_size = np.array(
            [
                int(np.round(images[0].shape[0] * padding_scale / 4) * 4),
                int(np.round(images[0].shape[1] * padding_scale / 4) * 4),
            ]
        )
        self.num_images = len(scan_angles)
        self.scan_lines = np.zeros(
            (images[0].shape[0], images[0].shape[1], self.num_images)
        )
        self.time_inds = np.linspace(
            -0.5, 0.5, self.scan_lines.shape[0]
        )  # Indices used for time-dependent linear drift calculation.
        self.w2 = np.zeros(self.image_size)  # Hanning window for hybrid correlation
        self.scan_origin = np.zeros((images[0].shape[0], 2, self.num_images))
        self.linear_search = (
            np.linspace(-0.04, 0.04, 1 + 2 * linear_search_steps)
            * self.scan_lines.shape[0]
        )

        self.scan_direction = np.zeros((self.num_images, 2))
        self.image_transform = np.zeros(
            (self.image_size[0], self.image_size[1], self.num_images)
        )
        self.image_density = np.zeros(
            (self.image_size[0], self.image_size[1], self.num_images)
        )
        self.linear_drift = None
        self.ref_point = self.image_size // 2
        # Load images into sMerge.scan_lines
        for i, img in enumerate(images):
            self.scan_lines[:, :, i] = img
            self.calculate_scan_origins(i, img)

        # Prepare for the linear drift search

    def run_linear_alignment(self):
        self.search_linear()
        self.refine_linear()
        self.apply_linear()
        self.init_alignment()
        self.show_alignment()

    def calculate_scan_origins(self, index, image):
        """
        Calculate scan origins based on the image and its scan angle.
        """
        height, width = image.shape
        xy = np.zeros((height, 2))
        xy[:, 0] = np.arange(1, height + 1) - height / 2
        xy[:, 1] = 1 - width / 2

        angle_rad = np.radians(self.scan_angles[index])
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[c, -s], [s, c]])
        xy_rotated = xy @ rotation_matrix.T
        xy_rotated[:, 0] += self.image_size[0] / 2
        xy_rotated[:, 1] += self.image_size[1] / 2
        xy_rotated[:, 0] -= np.mod(xy_rotated[0, 0], 1)
        xy_rotated[:, 1] -= np.mod(xy_rotated[0, 1], 1)
        xy_rotated = np.round(xy_rotated) - 1

        # Store scan origins
        self.scan_origin[:, :, index] = xy_rotated

        # Calculate the scan direction
        self.scan_direction[index, :] = [
            np.cos(angle_rad + np.pi / 2),
            np.sin(angle_rad + np.pi / 2),
        ]

    def search_linear(self):
        """
        Perform a search for linear drift vectors between images.
        """

        y_drift, x_drift = np.meshgrid(self.linear_search, self.linear_search)
        self.linear_search_score_1 = np.zeros_like(x_drift)

        # Create the hanning window
        N = self.scan_lines.shape[:2]
        window = self.hanning_local(N[0])[:, None] * self.hanning_local(N[1])[None, :]
        self.w2[
            (self.image_size[0] - N[0]) // 2 : (self.image_size[0] + N[0]) // 2,
            (self.image_size[1] - N[1]) // 2 : (self.image_size[1] + N[1]) // 2,
        ] = window

        for i, j in tqdm(
            np.ndindex(x_drift.shape),
            total=x_drift.size,
            desc="Initial Linear Drift Search",
        ):
            xy_shift = np.stack(
                [x_drift[i, j] * self.time_inds, y_drift[i, j] * self.time_inds],
                axis=-1,
            )

            # Apply linear drift to the first two images
            self.scan_origin[:, :, :2] += xy_shift[:, :, None]
            self.generate_trial_images(0)
            self.generate_trial_images(1)

            # Measure alignment score using hybrid correlation
            m = fft2d(self.w2 * self.image_transform[:, :, 0]) * np.conj(
                fft2d(self.w2 * self.image_transform[:, :, 1])
            )

            correlation_matrix = np.abs(
                ifft2d(np.sqrt(np.abs(m)) * np.exp(1j * np.angle(m)))
            )

            self.linear_search_score_1[i, j] = np.max(correlation_matrix)

            # Remove linear drift
            self.scan_origin[:, :, :2] -= xy_shift[:, :, None]

    def refine_linear(self):
        """
        Perform the second linear alignment to refine possible linear drift vectors.

        Parameters:
        - flag_report_progress: Boolean flag to enable or disable progress reporting.
        """
        # Find the maximum score from the first linear search
        max_index = np.argmax(self.linear_search_score_1)
        x_ind, y_ind = np.unravel_index(max_index, self.linear_search_score_1.shape)

        # Determine step size for refinement
        step = self.linear_search[1] - self.linear_search[0]

        # Refine the search grid around the best initial guess
        x_refine = (
            self.linear_search[x_ind]
            + np.linspace(-0.5, 0.5, len(self.linear_search)) * step
        )
        y_refine = (
            self.linear_search[y_ind]
            + np.linspace(-0.5, 0.5, len(self.linear_search)) * step
        )
        y_drift, x_drift = np.meshgrid(y_refine, x_refine)

        # Initialize score matrix for the second linear search
        self.linear_search_score_2 = np.zeros_like(self.linear_search_score_1)

        # Loop over all refined search grid points
        for a0, a1 in tqdm(
            np.ndindex(x_drift.shape),
            total=x_drift.size,
            desc="Refined Linear Drift Search",
        ):
            # Calculate time-dependent linear drift shift
            xy_shift = np.column_stack(
                (self.time_inds * x_drift[a0, a1], self.time_inds * y_drift[a0, a1])
            )

            # Apply linear drift to the first two images
            self.scan_origin[:, :, :2] += xy_shift[:, :, None]

            # Generate trial images for the first two images
            self.generate_trial_images(0)
            self.generate_trial_images(1)

            # Measure alignment score using hybrid correlation
            m = fft2d(self.w2 * self.image_transform[:, :, 0]) * np.conj(
                fft2d(self.w2 * self.image_transform[:, :, 1])
            )

            correlation_matrix = np.abs(
                ifft2d(np.sqrt(np.abs(m)) * np.exp(1j * np.angle(m)))
            )

            self.linear_search_score_2[a0, a1] = np.max(correlation_matrix)

            # Remove linear drift from the first two images
            self.scan_origin[:, :, :2] -= xy_shift[:, :, None]

        # Find the maximum score from the second linear search
        max_index = np.argmax(self.linear_search_score_2)
        x_ind, y_ind = np.unravel_index(max_index, self.linear_search_score_2.shape)

        # Store the best refined drift values
        self.linear_drift = [x_drift[x_ind, y_ind], y_drift[x_ind, y_ind]]
        return self.linear_drift

    def apply_linear(self):
        """
        Apply the linear drift correction to the images.

        """
        # Apply the linear drift to all images
        xyshifts = np.column_stack(
            (
                self.time_inds * self.linear_drift[0],
                self.time_inds * self.linear_drift[1],
            )
        )
        for i in range(self.num_images):
            self.scan_origin[:, :, i] += xyshifts
            self.generate_trial_images(i)

    def init_alignment(self):
        """
        Estimate the initial alignment between images.

        """
        # Apply linear drift correction
        dxy = np.zeros((self.num_images, 2))

        G1 = fft2d(self.w2 * self.image_transform[:, :, 0])

        # Loop over all pairs of images
        # for i in range(1, self.num_images):
        for i in tqdm(
            range(1, self.num_images),
            desc="Drift Correction (initial alignment) between images",
        ):
            G2 = fft2d(self.w2 * self.image_transform[:, :, i])
            # Calculate the alignment score
            m = G1 * np.conj(G2) / (np.abs(G1) * np.abs(G2))
            correlation_matrix = np.abs(
                ifft2d(np.sqrt(np.abs(m)) * np.exp(1j * np.angle(m)))
            )
            # Find the maximum correlation score
            max_index = np.argmax(correlation_matrix)
            x_ind, y_ind = np.unravel_index(max_index, correlation_matrix.shape)
            dx = x_ind - correlation_matrix.shape[0] // 2
            dy = y_ind - correlation_matrix.shape[1] // 2
            dxy[i, :] = dxy[i - 1, :] + [dx, dy]
            G2 = G1
        dxy = dxy - np.mean(dxy, axis=0)

        # Apply the alignment to all images
        for i in range(self.num_images):
            self.scan_origin[:, :, i] += dxy[i, :]
            self.generate_trial_images(i)

    def generate_trial_images(self, ind_image, ind_lines=None, plot=False):
        """
        This function generates a resampled scanning probe image with the dimensions
        of `image_size`, from an array of scan lines given in `scan_lines` (as rows).
        It also takes an array of Nx2 origins in `scan_or`, and a scan direction
        `scan_dir`, and stores all arrays in the struct `self`.

        Parameters:
        - ind_image: Index of the image to update.
        - ind_lines: Optional binary vector specifying which lines to include (default is all).
        """
        if ind_lines is None:
            ind_lines = np.ones(self.scan_lines.shape[0], dtype=bool)

        # Expand coordinates for resampling
        t = np.tile(np.arange(self.scan_lines.shape[1]), (np.sum(ind_lines), 1))
        x0 = np.tile(
            self.scan_origin[ind_lines, 0, ind_image], (self.scan_lines.shape[1], 1)
        ).T
        y0 = np.tile(
            self.scan_origin[ind_lines, 1, ind_image], (self.scan_lines.shape[1], 1)
        ).T

        x_ind = x0 + t * self.scan_direction[ind_image, 0]
        y_ind = y0 + t * self.scan_direction[ind_image, 1]

        # Prevent pixels from leaving image boundaries
        x_ind = np.clip(x_ind, 1, self.image_size[0] - 1)
        y_ind = np.clip(y_ind, 1, self.image_size[1] - 1)

        # Convert to bilinear interpolants and weights
        x_ind_floor = np.floor(x_ind).astype(int)
        y_ind_floor = np.floor(y_ind).astype(int)

        x_all = np.column_stack(
            [x_ind_floor, x_ind_floor + 1, x_ind_floor, x_ind_floor + 1]
        )
        y_all = np.column_stack(
            [y_ind_floor, y_ind_floor, y_ind_floor + 1, y_ind_floor + 1]
        )

        dx = x_ind - x_ind_floor
        dy = y_ind - y_ind_floor
        # stack the weights in a new axis
        weights = np.stack(
            [
                (1 - dx) * (1 - dy),  # top-left weight
                dx * (1 - dy),  # top-right weight
                (1 - dx) * dy,  # bottom-left weight
                dx * dy,
            ],  # bottom-right weight
            axis=-1,
        )  # Stack along the last axis
        ind_all = np.ravel_multi_index(
            (x_all.flatten(), y_all.flatten()), self.image_size
        )

        # Generate the image by accumulating the contributions
        scan_lines = self.scan_lines[ind_lines, :, ind_image]
        sig = np.zeros(np.prod(self.image_size))
        np.add.at(
            sig,
            ind_all,
            np.column_stack(
                [weights[:, :, i] * scan_lines for i in range(4)]
            ).flatten(),
        )
        sig = sig.reshape(self.image_size)

        count = np.zeros(np.prod(self.image_size))
        np.add.at(
            count,
            ind_all,
            np.column_stack([weights[:, :, i] for i in range(4)]).flatten(),
        )
        count = count.reshape(self.image_size)

        # Apply KDE (Kernel Density Estimate)
        r = max(int(np.ceil(self.kde_sigma * 3)), 5)
        sig = gaussian_filter(sig, self.kde_sigma, radius=r)
        count = gaussian_filter(count, self.kde_sigma, radius=r)

        valid_mask = count > 0
        sig[valid_mask] = sig[valid_mask] / count[valid_mask]

        self.image_transform[:, :, ind_image] = sig

        # Estimate sampling density
        boundary_mask = count == 0
        boundary_mask[[0, -1], :] = True
        boundary_mask[:, [0, -1]] = True

        # Distance calculation for boundary mask
        dist_to_boundary = distance_transform_edt(~boundary_mask)
        self.image_density[:, :, ind_image] = (
            np.sin(np.minimum(dist_to_boundary / self.edge_width, 1) * np.pi / 2) ** 2
        )
        if plot:
            plt.imshow(sig, cmap="gray")
            plt.savefig("image_transform.png")

    def hanning_local(self, N):
        """
        Replacement for 1D hanning function to avoid dependency.
        """
        return np.sin(np.pi * (np.arange(1, N + 1) / (N + 1))) ** 2

    def show_alignment(self):
        """
        Plot the final merged image with scanline origins overlaid.
        """
        # Plot the merged image and scanline origins
        # plt.imshow(np.mean(self.image_transform, axis=2), cmap='gray')
        image_avg = np.mean(self.image_transform, axis=2)
        density = np.prod(self.image_density, axis=2)
        mask = density > 0.5
        image_avg[~mask] = 0
        vmin = np.percentile(image_avg[mask], 1)
        plt.imshow(image_avg, cmap="gray", vmin=vmin)
        plt.scatter(
            self.ref_point[1],
            self.ref_point[0],
            c="r",
            s=50,
            label="Reference",
            marker="+",
        )

        for i in range(self.num_images):
            plt.scatter(
                self.scan_origin[:, 1, i],
                self.scan_origin[:, 0, i],
                label=f"Image {i}: scan {self.scan_angles[i]}°",
                s=1,
            )
        plt.legend()
        plt.show()
