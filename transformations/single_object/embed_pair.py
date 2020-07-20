import random
import numpy as np


def embed_pair(img, mask, back, added_width):
    if img.shape == back.shape:
        back[mask == 255] = img[mask == 255]
        back_mask = mask
    else:
        field_height = back.shape[0] - img.shape[0]
        field_width = back.shape[1] - img.shape[1] - added_width

        # Randomly choose point for left upper corner
        h = int(random.random() * field_height)
        w = int(random.random() * field_width) + int(added_width / 2)

        back[h:h + img.shape[0], w:w + img.shape[1], :][mask == 255] = img[mask == 255]
        back_mask = np.zeros(back.shape[:2]).astype('uint8')
        back_mask[h:h + img.shape[0], w:w + img.shape[1]][mask == 255] = mask[mask == 255]

    return back, back_mask
