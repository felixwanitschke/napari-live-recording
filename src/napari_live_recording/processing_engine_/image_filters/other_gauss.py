import numpy as np
from scipy import ndimage

parametersDict = {}

parametersHints = {}

functionDescription = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin at molestie tortor, tempus suscipit felis. Maecenas imperdiet ultricies urna, aliquet mollis lorem varius vitae. Nulla a nisi neque. Ut at turpis feugiat, tincidunt est sit amet, imperdiet magna. Pellentesque et viverra eros. Phasellus tempor turpis vulputate lacus lobortis mattis. Donec."


def blur(a):
    kernel = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])
    kernel = kernel / np.sum(kernel)
    arraylist = []
    for y in range(3):
        temparray = np.copy(a)
        temparray = np.roll(temparray, y - 1, axis=0)
        for x in range(3):
            temparray_X = np.copy(temparray)
            temparray_X = np.roll(temparray_X, x - 1, axis=1) * kernel[y, x]
            arraylist.append(temparray_X)

    arraylist = np.array(arraylist)
    arraylist_sum = np.sum(arraylist, axis=0)
    return arraylist_sum
