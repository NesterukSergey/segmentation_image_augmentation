def mask2bbox(mask):
    horizontal = mask.sum(axis=0)
    vertical = mask.sum(axis=1)
    x_min, x_max, y_min, y_max = None, None, None, None

    for i in range(len(horizontal)):
        if horizontal[i] > 0:
            x_min = i
            break

    for i in range(len(horizontal) - 1, -1, -1):
        if horizontal[i] > 0:
            x_max = i
            break

    for i in range(len(vertical)):
        if vertical[i] > 0:
            y_min = i
            break

    for i in range(len(vertical) - 1, -1, -1):
        if vertical[i] > 0:
            y_max = i
            break

    return [(x_min, y_max), (x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
