import numpy as np
import matplotlib.pylab as plt

# list args of the filter function and their desired default values in the parameterDict

shading_model = plt.imread('src/napari_live_recording/processing_engine_/image_filters/default_images/psf.png') 
parametersDict = {"parameter1": shading_model}

# give parameter hints for every parameter in paametersDict. This should contain a description as well as a hint which values (like:possible range, even/uneven numbers, parameter1 has to be larger than parameter2 ...) are allowed and which data-type is required.
parametersHints = {
    "parameter1": "Microscopes flatfield image. Default is the point spread function (PSF)"
}

# give a description of the function
functionDescription = "Recovery of the approx. true image by applying an inverse operation of the OTF in Fourier space. "

def inverse_filter(image, shading_model):
    """
    Also called 1/OT deconvolutionm, becaues it tryes to undo the effect of the convolution from the ilumination and the real image,
    by appling the inverse operation. This is done by deviding the fourier transform of the image by the fourier transform of the shading model.

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

    sharpened = ft(image) * w
    sharpened = np.abs(ift(sharpened))
    sharpened /= sharpened.max()

    return sharpened

def ft(im):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(im)))

def ift(im):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(im)))



