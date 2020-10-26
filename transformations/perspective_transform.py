import numpy as np
import cv2


def perspective_transform(scene, added_width):
    if added_width == 0:
        raise UserWarning('perspective_transform only needed if added_width > 0')

    offset = int(added_width / 2)

    matrix = cv2.getPerspectiveTransform(
        np.float32([[0, scene.shape[0]], [0, 0], [scene.shape[1], 0], [scene.shape[1], scene.shape[0]]]),
        np.float32([[0, scene.shape[0]], [offset, 0], [scene.shape[1] - offset, 0], [scene.shape[1], scene.shape[0]]])
    )

    transformed_background = cv2.warpPerspective(scene, matrix, (scene.shape[1], scene.shape[0]))

    if len(scene.shape) == 3:
        transformed_background = transformed_background[:, offset:-offset, :]
    else:
        transformed_background = transformed_background[:, offset:-offset]

    return transformed_background
