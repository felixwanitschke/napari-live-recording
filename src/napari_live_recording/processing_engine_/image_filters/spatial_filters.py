import skimage
import skimage.io
import skimage.exposure

import scipy

parametersDict = {"shape": (3, 3), "sigma": 0.5}

parametersHints = {"shape": "tuple,uneven numbers", "sigma": "float"}

def scipy_style_uniform(image, size):
    '''(linear filter)'''
    image = scipy.ndimage.filters.uniform_filter(image, size)
    return image

def scipy_style_gauss_2D(image, sigma):
    '''(linear filter)'''
    image = scipy.ndimage.filters.gaussian_filter(image, sigma)
    return image

def scipy_style_median(image, size):
    image = scipy.ndimage.filters.median_filter(image, size)
    return image

def skimage_style_gauss_2D(image, sigma):
    image = skimage.filters.gaussian(image, sigma)
    return image

def skimage_style_sobel(image):
    image = skimage.filters.sobel(image)
    return image

def skimage_style_median(image, size):
    '''The median filter preserves edges and can be used to romove isolated noise pixels. (nonlinear filter)
    args:
        image: numpy array
        size: int
    retruns:
        image: numpy array
    '''
    image = skimage.filters.median(image, size)
    return image