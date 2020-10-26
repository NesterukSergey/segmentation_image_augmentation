from transformations import *


def transform_single_object(img, mask,
                            output_size_mode='fixed', output_size=[1080, 1920], output_scale=2,
                            max_rotate_degree=30, flip_prob=0.5,
                            move=True,
                            background='none', background_image=None,
                            persp_trans=0,
                            salt=0.1, pepper=0.1, gauss_var=0.1,
                            smooth_kernel_size=True):
    """Applies transformations to image and mask.

    Args:
        img (numpy.array): initial RGB image with shape (height, width, 3).
        mask (numpy.array): binary mask for initial image. Must match the size of img at first 2 dimensions.
                            Can be RGB or grayscale. '1' for the object, '0' for the background.
        output_size_mode (str): the rule to define output size.
            'fixed' to set size manually, requires output_size.
            'multiplier' to set relation between input and output size, requires output_scale.
        output_size (list): defines [height, width] of output image if output_size_mode is 'fixed'.
        output_scale: sets the scale of output size based on input size if output_size_mode is 'multiplier'.
            (float): Multiplier for both axes. Should be >= 1.
            (tuple): Multipliers for separate axis. Both should be >= 1.
        max_rotate_degree (int): sets maximum degrees of random rotation.
        flip_prob (float): the probability of horizontal flip.
        move (bool): if True, the object will be randomly moved.
        background (str): mode of adding background.
            'none' for white background.
            'img' to provide custom background image.
        background_image (numpy.array): if background is 'img', provide an RGB image for the background.
        persp_trans (float): the percent of image width that will be added to initial image to perform perspective transform.
                             '0' means no transform.
        salt (float): the probability to make each of the img pixels white.
        pepper (float): the probability to make each of the img pixels black.
        gauss_var (float): the variance of applied additive Gauss noise.
        smooth_kernel_size (int): size og Gauss smoothing kernel. Must be an odd number.

    Returns:
        transformed_img (numpy.array), transformed_mask (numpy.array)
        bbox (list of tuples): points of object's bounding box.
    """
    if img.shape[:2] != mask.shape[:2]:
        raise UserWarning('Mask size must match image size')

    if ((img.shape[2] != 3)
            or (len(mask.shape) == 3 and mask.shape[2] != 3)):
        raise NotImplementedError('Only RGB image supported')

    if output_size_mode == 'fixed':
        scene_shape = output_size
    elif output_size_mode == 'multiplier':
        if 'tuple' in str(type(output_scale)):
            scene_shape = (int(img.shape[0] * output_scale[0]), img.shape[1] * output_scale[1], 3)
        else:
            scene_shape = (int(img.shape[0] * output_scale), img.shape[1] * output_scale, 3)
    else:
        raise UserWarning('Wrong output_size_mode value')

    added_width = 0
    if background == 'img':
        if persp_trans < 0:
            raise UserWarning('persp_trans must be >= 0')

        if persp_trans > 0:
            added_width = int(scene_shape[1] * persp_trans)
            added_width = added_width if added_width % 2 == 0 else added_width + 1
            scene = get_background(background_image, scene_shape, added_width)
        else:
            added_width = 0
            scene = resize(background_image, scene_shape)
    else:
        scene = np.ones((scene_shape[0], scene_shape[1], 3)).astype('uint8')

    mask = binarize_mask(mask)

    img, mask = flip_pair(img, mask, flip_prob)
    img, mask = rotate_pair(img, mask, max_rotate_degree)

    scene, mask = embed_pair(img, mask, scene, added_width)

    if added_width > 0:
        scene = perspective_transform(scene, added_width)
        mask = perspective_transform(mask, added_width)

    scene = add_salt(scene, salt)
    scene = add_pepper(scene, pepper)
    scene = gauss_noise(scene, gauss_var)
    scene = smooth(scene, smooth_kernel_size)

    bbox = mask2bbox(mask)

    return scene, mask, bbox
