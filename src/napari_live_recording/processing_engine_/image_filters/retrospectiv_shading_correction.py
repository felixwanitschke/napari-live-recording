import numpy as np
import skimage

# list args of the filter function and their desired default values in the parameterDict
snr = 1e4
box = [((34, 31), (45, 48)), # upper left
                ((285,18), (300,40)), # lower left
                ((15, 220), (28, 240)), # upper right
                ((280,225), (290,245))] # lower right
parametersDict = {"parameter1": snr, "parameter2": box}

# give parameter hints for every parameter in paametersDict. This should contain a description as well as a hint which values (like:possible range, even/uneven numbers, parameter1 has to be larger than parameter2 ...) are allowed and which data-type is required.
parametersHints = {
    "parameter1": "signal-to-noise ratio",
    "parameter2": "boxes in the image which contain mainly background signal, list of tuples, each tuple contains two tuples with the coordinates of the lower left and upper right corner of the box.",
}

# give a description of the function
functionDescription = "Technique to correct for non-uniform illumination artifacts in images. Instead of acquiring the dark images and flat field images, we can use the image itself to estimate the background field and the effective illumination."

def retrospective_shadng_correction(image, threashold1, threashold2):
            
    ''' Retrospective shading correction is a technique used to correct for non-uniform illumination artifacts in images. 
        Instead of acquiring the dark images and flat field images, we can use the image itself to estimate the background field 
        and the effective illumination.'''

    raw = image[:,:248]

    original = skimage.img_as_ubyte(image)
    #snr = 1e4
    sigma = np.sqrt(np.var(original) / threashold1)
    noise = np.random.randn(*original.shape) * sigma

    image = original + noise
    image -= image.min()
    image /= image.max()
    image = 0.5 * (image + 1)
    image = skimage.img_as_ubyte(image)

    # boxes containing mainly background signal
    # threashold2(boxes) = [((34, 31), (45, 48)), # upper left
    #         ((285,18), (300,40)), # lower left
    #         ((15, 220), (28, 240)), # upper right
    #         ((280,225), (290,245))] # lower right

    coords = []
    for box in threashold2:
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