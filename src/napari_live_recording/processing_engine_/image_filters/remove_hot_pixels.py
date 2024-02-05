import numpy as np

# list args of the filter function and their desired default values in the parameterDict
pixel_value = 20

parametersDict = {"parameter1": pixel_value}

# give parameter hints for every parameter in paametersDict. This should contain a description as well as a hint which values (like:possible range, even/uneven numbers, parameter1 has to be larger than parameter2 ...) are allowed and which data-type is required.
parametersHints = {
    "parameter1": "signal-to-noise ratio"
}

# give a description of the function
functionDescription = "Remove hot pixels from an image by replacing them with the average of the 4 neighboring pixels"

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
        is_connected = False

        # Check if the current hot pixel is connected to any other hot pixel
        for j in range(len(hot_pixels[0])):
            if i != j and are_neighbors((x, y), (hot_pixels[0][j], hot_pixels[1][j])):
                is_connected = True

        # If the current hot pixel is not connected to any other hot pixel, smooth it out
        if not is_connected:
            image[x,y] = (image[x-1,y] + image[x+1,y] + image[x,y-1] + image[x,y+1])/4

    return image  

def are_neighbors(pixel1, pixel2):
    x1, y1 = pixel1
    x2, y2 = pixel2

    return (x1 == x2 and abs(y1 - y2) == 1) or (y1 == y2 and abs(x1 - x2) == 1)