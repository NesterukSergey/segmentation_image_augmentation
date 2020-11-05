import copy
import math
import numpy as np

# import rpack
from rectpack import newPacker
from rectpack.maxrects import MaxRectsBssf


def _change_dim_order(sizes):
    return [[s[1], s[0]] for s in sizes]


# def get_pack_coords(sizes):
#     # list of [height, width] i.e. img.shape order
#     sizes = _change_dim_order(sizes)
#     positions = rpack.pack(sizes)
#     return _change_dim_order(positions)


def _pack(rectangles, bins):
    packer = newPacker(pack_algo=MaxRectsBssf)

    for r in rectangles:
        packer.add_rect(*r)

    for b in bins:
        packer.add_bin(*b)

    packer.pack()

    all_rects = packer.rect_list()

    res = []
    for rect in all_rects:
        res.append(np.array(rect))

    res = np.array(res)
    res.view('i8,i8,i8,i8,i8,i8,').sort(order=['f5'], axis=0)
    res = [list(i) for i in res[:, 1:3]]
    return res


def get_pack_coords(sizes):
    s = copy.deepcopy(sizes)
    [s[i].append(i + 1) for i in range(len(s))]
    s = np.array([np.array(i) for i in s]).copy()

    total_h, total_w, _ = s.sum(axis=0)
    max_h = s[:, 0].max(axis=0)

    virtual_cols = math.ceil(math.sqrt(len(sizes)))
    height_limit = max(max_h, int(1.2 * (total_h / virtual_cols)))

    rectangles = [tuple(i) for i in s]
    bins = [(height_limit, total_w)]

    coords = _pack(rectangles, bins)

    if len(coords) != len(sizes):
        coords = _pack(rectangles, [(int(2 * max_h), total_w)])

    return coords
