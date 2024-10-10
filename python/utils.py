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
