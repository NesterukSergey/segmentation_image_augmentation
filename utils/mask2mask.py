import numpy as np
from utils.colors import generate_colors


def semantic2binary(mask):
    return (mask > 0).max(axis=2).astype('uint8') * 255


def single2multi(mask):
    return np.stack((mask,) * 3, axis=-1)


def semantic2binary_list(mask):
    """Input RGB image"""
    unsqueezed_mask = mask.reshape(-1, mask.shape[2])
    masks_colors = np.unique(unsqueezed_mask, axis=0)
    background_index = np.argwhere(np.sum(masks_colors, axis=1) == 0)
    masks_colors = np.delete(masks_colors, background_index, 0)
    colors_count = masks_colors.shape[0]

    masks = []
    for i in range(colors_count):
        masks.append((mask == masks_colors[i]).reshape(mask.shape).astype('uint8') * 255)

    return masks


def binary_list2semantic(mask_list):
    main_mask = np.zeros_like(mask_list[0])
    colors = generate_colors(len(mask_list))

    for i, mask in enumerate(mask_list):
        main_mask[:, :, :3][mask[:, :, 0] > 0] = colors[i]

    return main_mask.astype('uint8')


