import random


def random_crop(img, size):
    if (img.shape[0] < size[0]) or (img.shape[1] < size[1]):
        raise UserWarning("Can't crop a big image from a small one")

    if (img.shape[0] == size[0]) and (img.shape[1] == size[1]):
        return img

    start_height = random.randint(0, img.shape[0] - size[0] - 1)
    start_width = random.randint(0, img.shape[1] - size[1] - 1)

    return img[start_height:start_height + size[0], start_width:start_width + size[1], :]
