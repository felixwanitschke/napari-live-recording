import numpy as np
import matplotlib.pylab as plt
import scipy.signal as signal
import tkinter as tk
from tkinter import filedialog, ttk
import skimage

parametersDict = {"shape": (3, 3), "sigma": 0.5}

parametersHints = {"shape": "tuple,uneven numbers", "sigma": "float"}

def ift(im):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(im)))

def ft(im):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(im)))

def convolve(f, g):
    """Full convolution of `f` and `g` computed in real space. 

    args:
        - f (np.ndarray): The first argument is the signal.
        - g (np.ndarray): The second argument is the filter. 
    It is assumed that the size of the signal is at least the size of the filter.
    
    Returns:
        - Full convolution stored in an array of size 'len(f) + len(g) - 1'
    """
    N, M = len(f), len(g)
    assert N >= M
    # reverse filter
    g = g[::-1]
    # pad zeros to signal
    f = np.pad(f, M-1)
    return np.array([f[i:i+M] @ g for i in range(N+M-1)])

def fftconvolve(f,g):
    '''Full convolution of `f` and `g` computed with FFT'''

    size = len(f) + len(g) -1 #size of full convolution

    F= np.fft.fft(f,size)
    G= np.fft.fft(g,size)

    return np.fft.ifft(F*G)



def remove_hot_pixels(image, hotvalue):
    '''Remove hot pixels from an image by replacing them with the average of the 4 neighboring pixels.
    Args:
        - image: numpy array
        - hotvalue: int
    Returns:
        - image: numpy array
    '''
    # find hot pixels
    hot_pixels = np.where(image == hotvalue)

    # replace hot pixels with average of 4 neighboring pixels
    for i in range(len(hot_pixels[0])):
        x = hot_pixels[0][i]
        y = hot_pixels[1][i]
        image[x,y] = (image[x-1,y] + image[x+1,y] + image[x,y-1] + image[x,y+1])/4

    return image  

def inverse_filter(raw_image, shading_model):
    """
    Apply inverse filtering to the raw image using the flat field correction.

    Parameters:
    raw_image (ndarray): The raw input image.
    flat_field_correction (ndarray): The flat field correction image.

    Returns:
    ndarray: The sharpened image after inverse filtering.
    """
    otf = ft(shading_model)
    mtf = np.abs(otf)
    c = 1e-35
    
    w = otf.conj() / (np.square(mtf) + c)

    sharpened = ft(raw_image) * w
    sharpened = np.abs(ift(sharpened))
    sharpened /= sharpened.max()

    return sharpened

def wiener_filter(raw_image, shading_model, k):

    npix = len(raw_image)
    radii = np.array([npix//2, npix//4, npix//16, npix//64])
    c = 1e-5
    gamma = 0.01

    for i, radius in enumerate(radii):
        otf = ft(shading_model)
        mtf = np.abs(otf)
        # Wiener filter
        c_work = c if 2*radius < npix else 0.
        ifilt = otf.conj() / (np.square(mtf) + c_work)
        filt = np.real(ift(ifilt))

        # apply filter
        if i == 0:
            filtered = filt * ft(raw_image)
        else:
            filtered = filt * ft(filtered)

    return filtered