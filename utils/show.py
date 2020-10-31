import matplotlib.pyplot as plt


def show(img, bbox=None):
    """Plots image (and image mask).

    Args:
        img (numpy.array): image.
            or (list) - list that consists og image and mask
    """
    if 'list' in str(type(img)):
        assert len(img) == 2

        fig, ax = plt.subplots(1, 2, figsize=(12, 12))
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        if len(img[0].shape) == 2:
            ax[0].imshow(img[0], cmap='gray', vmin=0, vmax=255)
        else:
            ax[0].imshow(img[0])

        if bbox is not None:
            for p in range(4):
                ax[0].plot([bbox[p][0], bbox[p + 1][0]], [bbox[p][1], bbox[p + 1][1]], 'r')

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


def show_line(images, bboxes=None):

    if bboxes is None:
        bboxes = ([None] * len(images))

    fig, ax = plt.subplots(1, len(images), figsize=(18, 10 * len(images)))

    for i in range(len(images)):
        ax[i].set_xticks([])
        ax[i].set_yticks([])

        ax[i].imshow(images[i])

        if bboxes[i] is not None:
            for bbox in bboxes[i]:
                for p in range(4):
                    ax[i].plot([bbox[p][0], bbox[p + 1][0]], [bbox[p][1], bbox[p + 1][1]], 'r')

    plt.show()
