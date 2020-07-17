import matplotlib.pyplot as plt


def read(path):
    """Reads an image from the specified path.
    ar
    Args:
        path (str): image path.

    :Returns:
        img (numpy.array): RGB image
    """
    return plt.imread(path)[:, :, :3]
