import numpy as np


def compute_diff(image):
    '''
    Parameters
    ----------
    image: numpy.ndarray of shape nxn
    Returns
    -------
    below: numpy.ndarray represents image[i][j]-image[i+1][j]
    to_right: numpy.ndarray represents image[i][j]-image[i][j+1]
    '''
    zips = np.zeros(image.shape[-1])[np.newaxis, :]
    above = np.concatenate([zips, image])
    below = np.concatenate([image, zips])

    below = above - below
    below = below[1:, :]
    below[-1, :] = 0

    zips = np.zeros(image.shape[-1])[:, np.newaxis]
    right_side = np.concatenate([image, zips], axis=1)
    left_side = np.concatenate([zips, image], axis=1)
    to_right = left_side - right_side
    to_right = to_right[:, 1:]
    to_right[:, -1] = 0
    return below, to_right