import scipy

# list args of the filter function and their desired default values in the parameterDict
size = 20

parametersDict = {"parameter1": size}

# give parameter hints for every parameter in paametersDict. This should contain a description as well as a hint which values (like:possible range, even/uneven numbers, parameter1 has to be larger than parameter2 ...) are allowed and which data-type is required.
parametersHints = {
    "parameter1": "The sizes of the uniform filter are given for each axis as a sequence, or as a single number, in which case the size is equal for all axes. int or sequence of ints, optional"
}

# give a description of the function
functionDescription = "Technique to correct for non-uniform illumination artifacts in images. Instead of acquiring the dark images and flat field images, we can use the image itself to estimate the background field and the effective illumination."


def scipy_style_uniform(image, threashold1):
    '''(linear filter)'''
    image = scipy.ndimage.filters.uniform_filter(image, threashold1)
    return image