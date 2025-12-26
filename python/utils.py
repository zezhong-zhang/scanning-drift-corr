import numpy as np

def fft2d(arr):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(arr)))

def ifft2d(arr):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(arr)))

def fft_shift_image(im, xshift, yshift):
    """
    Generates a shifted image using fft, with supports for subpixel shifts.

    Inputs:
        im      ndarray, shape=(nx,ny)  Image to shift
        xshift  int or float            pixels to shift x
        yshift  int or float            pixels to shift y

    Outputs:
        im_shifted  ndarray, shape=(nx,ny)  Shifted image
    """

    # Define real space meshgrids
    nx, ny = np.shape(im)
    rx, ry = np.mgrid[0:nx, 0:ny]

    w = -np.exp((2j * np.pi) * (xshift * rx / nx + yshift * ry / ny))
    im_shifted = np.abs(
        np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(im)) * w))
    )

    return im_shifted


def kde_convolve(arr, kernel):
    """
    Convolve a 2D array with a kernel using FFT.

    Inputs:
        arr     ndarray, shape=(nx,ny)  Image to convolve
        kernel  ndarray, shape=(nx,ny)  Kernel to convolve with

    Outputs:
        arr_convolved  ndarray, shape=(nx,ny)  Convolved image
    """

    # Get the Fourier transforms of the arrays
    arr_fft = fft2d(arr)
    kernel_fft = fft2d(kernel)

    # Convolve the arrays
    arr_convolved = ifft2d(arr_fft * kernel_fft)

    return np.abs(arr_convolved)

def calcScore(image, xF, yF, dx, dy, intMeas):
    """Bilinear interpolation and score calculation."""
    idx1 = np.ravel_multi_index((xF, yF), image.shape, mode='clip')
    idx2 = np.ravel_multi_index((xF + 1, yF), image.shape, mode='clip')
    idx3 = np.ravel_multi_index((xF, yF + 1), image.shape, mode='clip')
    idx4 = np.ravel_multi_index((xF + 1, yF + 1), image.shape, mode='clip')
    
    # Bilinear interpolation
    imageSample = (
        image.flat[idx1] * (1 - dx) * (1 - dy) +
        image.flat[idx2] * dx * (1 - dy) +
        image.flat[idx3] * (1 - dx) * dy +
        image.flat[idx4] * dx * dy
    )
    
    # Calculate score
    return np.sum(np.abs(imageSample - intMeas))


def hanning_local(n):
    """
    Replacement for 1D hanning function to avoid dependency.
    """
    return np.sin(np.pi * (np.arange(1, n + 1) / (n + 1))) ** 2

def make_fourier_coordinates(N, pSize):
    """
    Generates Fourier coordinates similar to the MATLAB function.
    
    Parameters:
    - N: Number of points
    - pSize: Pixel size
    
    Returns:
    - q: Array of Fourier coordinates
    """
    if N % 2 == 0:
        q = np.arange(-N/2, N/2) / (N * pSize)
        q = np.roll(q, -N//2)
    else:
        q = np.arange(-N/2 + 0.5, N/2 - 0.5 + 1) / ((N - 1) * pSize)
        q = np.roll(q, int(-N/2 + 0.5))
    return q
