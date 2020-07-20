import numpy as np
from transformations.single_object.resize import resize


def get_background(back, scene_shape, added_width):
    if back is None:
        raise UserWarning("background='img' requires background_image parameter")

    if len(back.shape) != 3:
        raise UserWarning('Only RGB background_image supported')

    back = resize(back, (scene_shape[0], scene_shape[1] + added_width))
    return back
