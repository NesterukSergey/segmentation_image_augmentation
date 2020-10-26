import itertools
import math
import numpy as np


def generate_colors(n):
    levels_count = math.ceil(math.pow(n + 2, 1 / 3))  # remove black and white
    step = 1 / levels_count
    levels = [1 - (step * i) for i in range(levels_count)]

    colors = []
    for p in itertools.product(levels, repeat=3):
        colors.append((np.array([*p]) * 255).astype(int))

    return colors[1:n + 1]
