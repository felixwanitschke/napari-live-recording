import numpy as np
import matplotlib.pylab as plt
import tkinter as tk
from tkinter import filedialog, ttk
import skimage

def select_dark_images(image):
    maxval = np.iinfo(image.dtype).max

    root = tk.Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    file_path = filedialog.askopenfilename()

    # load and average dark frames "background01.tif", "background02.tif", etc.
    dark = 0.
    for i in range(1, 6):
        dark += plt.imread(file_path + f'/background0{i}.tif') / maxval
    dark /= 5

    return dark

def select_flat_field_images(image):
    maxval = np.iinfo(image.dtype).max

    root = tk.Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    file_path = filedialog.askopenfilename()

    #load and average flat fields "flatfield01.tif", "flatfield02.tif", etc.
    flat = 0.
    for i in range(1, 5):
        flat += plt.imread(file_path + f'/flatfield0{i}.tif') / maxval
    flat /= 4

    return flat


def flat_field_correction(raw_image):
    '''Flat field correction is a technique used to correct for non-uniform illumination artifacts in images. Also know as flatfielding or shading correction.

    Args:
        - dark_image: averaged to estimate the background field
        - flat_field_image: measuring a full-field illumination image of a homogeneous fluorescent sample to determine the effective illumination.
            (by averaging we reduce the noise and obtain the average flat field image)
    '''
    maxval = np.iinfo(raw_image.dtype).max
    raw_image = raw_image / maxval

    # load the averaged dark frames
    dark_image = select_dark_images(raw_image)
    # load the averaged flat fields
    flat_field_image = select_flat_field_images(raw_image)

    # flat-field correction
    corr_image = (raw_image-dark_image)/(flat_field_image-dark_image)
    corr_image /= corr_image.max()

    return corr_image 

def retrospective_shadng_correction(image, snr=1e4, boxes = [((34, 31), (45, 48)), # upper left
                                                            ((285,18), (300,40)), # lower left
                                                            ((15, 220), (28, 240)), # upper right
                                                            ((280,225), (290,245))] # lower right
            ):
    ''' Retrospective shading correction is a technique used to correct for non-uniform illumination artifacts in images. 
        Instead of aquirreing the dark images and flat field images, we can use the image itself to estimate the background field 
        and the effective illumination.
        s(x, y) = s1(1 − x)(1 − y) + s2 x(1 − y) + s3(1 − x)y + s4 xy'''

    raw = image[:,:248]

    original = skimage.img_as_ubyte(image)
    #snr = 1e4
    sigma = np.sqrt(np.var(original) / snr)
    noise = np.random.randn(*original.shape) * sigma

    image = original + noise
    image -= image.min()
    image /= image.max()
    image = 0.5 * (image + 1)
    image = skimage.img_as_ubyte(image)

    # boxes containing mainly background signal
    # boxes = [((34, 31), (45, 48)), # upper left
    #         ((285,18), (300,40)), # lower left
    #         ((15, 220), (28, 240)), # upper right
    #         ((280,225), (290,245))] # lower right

    coords = []
    for box in boxes:
        lower, upper = box
        patch = raw[lower[0]:upper[0]+1, lower[1]:upper[1]+1]
        coords.append((0.5 * (lower[0] + upper[0]) / raw.shape[0],
        0.5 * (lower[1] + upper[1]) / raw.shape[1],
        patch.mean()))

    b = []
    A = []
    for x, y, z in coords:
        b.append(z)
        A.append([(1-x)*(1-y), x*(1-y), (1-x)*y, x*y])

    b = np.array(b) - raw.min()
    A = np.array(A)
    s = np.linalg.solve(A, b)

    y, x = np.mgrid[0:1:raw.shape[0]*1j, 0:1:raw.shape[1]*1j]
    estimated = s[0] * (1-x) * (1-y) \
                + s[1] * (1-x) * y \
                + s[2] * x * (1-y) \
                + s[3] * x * y
    
    esti_corr_image = image / estimated

    return esti_corr_image