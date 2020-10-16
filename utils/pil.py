import numpy as np
from PIL import Image


def np2pil(img):
    return Image.fromarray((img * 255).astype('uint8'))


def pil2np(img):
    np_img = np.asarray(img)

    if np_img.max() <= 1:
        return (np_img * 255).astype('uint8')
    else:
        return np_img
