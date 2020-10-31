import numpy as np
from PIL import Image

from utils.format_image import format_image


def np2pil(img):
    i = img.copy()
    return Image.fromarray(format_image(i))


def pil2np(img):
    np_img = np.asarray(img)
    return format_image(np_img)
