import numpy as np


def one_hot(array):
    number_of_classes = np.max(array) + 1
    return np.eye(number_of_classes, dtype=np.int32)[array]


def flatten(image):
    pixels, shape = 1, image.shape[1:]

    for size in shape:
        pixels *= size

    flattened = np.reshape(image, (-1, pixels))
    return flattened / 255
