import random
from utils.pil import *


def rotate_pair(img, mask, degree):
    """Rotates image and mask for the degree from range [-degree, degree]

    Args:
        img (numpy.array): RGB or grayscale image.
        mask (numpy.array): image mask.
        degree (int): maximum rotation degree.

    Returns:
        img (numpy.array): rotated image.
        mask (numpy.array): rotated image mask.
    """
    rotation_degree = int(random.random() * degree)
    rotation_degree = rotation_degree if random.random() < 0.5 else -rotation_degree

    img = rotate(img, rotation_degree)
    mask = rotate(mask, rotation_degree)

    return img, mask


def rotate(img, degree):
    """Rotates image for the specified degree.

    Args:
        img (numpy.array): RGB or grayscale image.
        degree (int): rotation degree

    Returns:
        img (numpy.array): rotated image.
    """
    pil_img = np2pil(img)
    pil_img = pil_img.rotate(degree)
    np_img = pil2np(pil_img)

    return np_img
