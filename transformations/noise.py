import numpy as np
import cv2


def add_salt(img, p):

    if p == 0:
        return img

    img_copy = img.copy()
    salt_mask = np.random.rand(*img.shape[:2])
    salt_mask = salt_mask < p
    salt = (np.ones(img.shape) * 255).astype('uint8')
    img_copy[salt_mask] = salt[salt_mask]

    return img_copy


def add_pepper(img, p):

    if p == 0:
        return img

    img_copy = img.copy()
    pepper_mask = np.random.rand(*img.shape[:2])
    pepper_mask = pepper_mask < p
    pepper = (np.zeros(img.shape) * 255).astype('uint8')
    img_copy[pepper_mask] = pepper[pepper_mask]

    return img_copy


def gauss_noise(img, var):

    if var == 0:
        return img

    gauss = (np.random.normal(0, var**0.5, img.shape) * 255).astype('uint8')
    img = img + gauss

    return np.clip(img, 0, 255)


def smooth(img, kernel_size):

    if kernel_size == 1:
        return img

    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
