import cv2


def resize(img, shape):
    return cv2.resize(img, (shape[1], shape[0]))
