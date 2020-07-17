import matplotlib.pyplot as plt


def show(img):
    """Plots image (and image mask).

    Args:
        img (numpy.array): image.
            or (list) - list that consists og image and mask
    """
    if 'list' in str(type(img)):
        assert len(img) == 2
        assert img[0].shape[:2] == img[1].shape[:2]

        fig, ax = plt.subplots(1, 2)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        if len(img[0].shape) == 2:
            ax[0].imshow(img[0], cmap='gray', vmin=0, vmax=255)
        else:
            ax[0].imshow(img[0])

        ax[1].set_xticks([])
        ax[1].set_yticks([])
        if len(img[1].shape) == 2:
            ax[1].imshow(img[1], cmap='gray', vmin=0, vmax=255)
        else:
            ax[1].imshow(img[1])

        plt.show()
    else:
        plt.xticks([])
        plt.yticks([])

        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        else:
            plt.imshow(img)

        plt.show()
