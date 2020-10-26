def binarize_mask(mask):
    """Returns binary 2D mask.

    Args:
        mask (numpy.array): RGB or grayscale mask.

    Returns:
        mask (numpy.array): binarized grayscale mask.
    """
    if len(list(mask.shape)) == 2:
        return mask
    else:
        return (mask > 0).max(axis=2).astype(int) * 255
