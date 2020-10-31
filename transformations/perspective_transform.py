import numpy as np
import cv2


def perspective_transform(scene, added_width):
    if added_width == 0:
        raise UserWarning('perspective_transform only needed if added_width > 0')

    offset = int(added_width / 2)

    matrix = cv2.getPerspectiveTransform(
        np.float32([[0, 0],      [scene.shape[1], 0],          [scene.shape[1], scene.shape[0]], [0, scene.shape[0]]]),
        np.float32([[offset, 0], [scene.shape[1] - offset, 0], [scene.shape[1], scene.shape[0]], [0, scene.shape[0]]])
    )

    transformed_background = cv2.warpPerspective(scene, matrix, (scene.shape[1], scene.shape[0]))

    if len(scene.shape) == 3:
        transformed_background = transformed_background[:, offset:-offset, :]
    else:
        transformed_background = transformed_background[:, offset:-offset]

    return transformed_background


def bbox_perspective_transform(bbox, added_width, scene_size):
    if added_width == 0:
        raise UserWarning('perspective_transform only needed if added_width > 0')

    offset = int(added_width / 2)

    matrix = cv2.getPerspectiveTransform(
        np.float32([[0, 0],      [scene_size[1], 0],          [scene_size[1], scene_size[0]], [0, scene_size[0]]]),
        np.float32([[offset, 0], [scene_size[1] - offset, 0], [scene_size[1], scene_size[0]], [0, scene_size[0]]])
    )

    [[(x_min1, y_max1), (x_min2, y_min1), (x_max1, y_min2), (x_max2, y_max2), (_, _)]] = cv2.perspectiveTransform(bbox, matrix).astype(int)
    x_min = min(x_min1, x_min2)
    y_min = min(y_min1, y_min2)
    x_max = max(x_max1, x_max2)
    y_max = max(y_max1, y_max2)

    return [(x_min, y_max), (x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    # return cv2.perspectiveTransform(bbox, matrix).astype(int)[0]
