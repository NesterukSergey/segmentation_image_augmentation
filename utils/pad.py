import numpy as np


def pad(img, border):
    return np.stack([np.pad(img[:, :, c], border, mode='constant', constant_values=0) for c in range(3)], axis=2)
