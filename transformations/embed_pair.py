import math


# def embed_pair(img, mask, back, added_width=0, start_coords=None):
#     if img.shape == back.shape:
#         back[mask == 255] = img[mask == 255]
#         back_mask = mask
#     else:
#         field_height = back.shape[0] - img.shape[0]
#         field_width = back.shape[1] - img.shape[1] - added_width
#
#         if start_coords is not None:
#             h, w = start_coords
#         else:
#             # Randomly choose point for left upper corner
#             h = int(random.random() * field_height)
#             w = int(random.random() * field_width) + int(added_width / 2)
#
#         back[h:h + img.shape[0], w:w + img.shape[1], :][mask == 255] = img[mask == 255]
#         back_mask = np.zeros(back.shape[:2]).astype('uint8')
#         back_mask[h:h + img.shape[0], w:w + img.shape[1]][mask == 255] = mask[mask == 255]
#
#     return back, back_mask


def embed_pair(img, mask, back, main_mask, added_width=0, start_coords=None):

    h, w = start_coords
    width_start = math.floor(added_width / 2)

    back[h:h + img.shape[0], w + width_start:w + img.shape[1] + width_start, :][mask > 0] = img[mask > 0]
    main_mask[h:h + img.shape[0], w + width_start:w + img.shape[1] + width_start, :][mask > 0] = mask[mask > 0]

    return back, main_mask
