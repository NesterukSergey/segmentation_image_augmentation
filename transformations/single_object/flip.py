import random


def flip_pair(img, mask, p=1.0):
    """Flips image and mask horizontally with probability p.

    Args:
        img (numpy.array): RGB or grayscale image.
        mask (numpy.array): object image.
        p (float): probability of flipping.

    Returns:
        img (numpy.array): flipped image.
        mask (numpy.array): flipped object image.
    """
    if random.random() < p:
        img = flip(img)
        mask = flip(mask)

    return img, mask


def flip(img):
    """Flips image horizontally.

    Args:
        img (numpy.array): RGB or grayscale image.

    Returns:
        img (numpy.array): flipped image.
    """
    if len(img.shape) == 3:
        return img[:, ::-1, :]
    else:
        return img[:, ::-1]
