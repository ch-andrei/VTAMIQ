import numpy as np


def normalize_array(a):
    b = a - np.min(a)
    if abs(b.max()) > 1e-6:
        b /= b.max()
    return b


# converts an image to an array of color values of shape 3xN where N is the number of pixels
def img2array(img, c=3, cast_float=True):
    w, h = img.shape[:2]
    ar = img.flatten().reshape((w * h, c)).transpose()
    return ar.astype(np.float32) if cast_float else ar


def ensure3d(ar):
    ar = np.array(ar)
    if len(ar.shape) == 2:
        ar = np.atleast_3d(ar)
        ar = np.repeat(ar, 3, axis=2)
    return ar
