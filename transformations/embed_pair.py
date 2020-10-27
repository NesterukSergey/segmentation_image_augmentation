import math


def embed_pair(img, mask, back, small_masks_list, main_masks_list, added_width=0, start_coords=None):
    h, w = start_coords
    width_start = math.floor(added_width / 2)

    m = (mask[:, :, 0] > 0) | (mask[:, :, 1] > 0) | (mask[:, :, 2] > 0)

    back[h:h + img.shape[0], w + width_start:w + img.shape[1] + width_start, :][m] = img[m]

    for mask_type in main_masks_list:
        main_masks_list[mask_type][h:h + img.shape[0], w + width_start:w + img.shape[1] + width_start, :][m] = \
        small_masks_list[mask_type][m]

    return back, main_masks_list
