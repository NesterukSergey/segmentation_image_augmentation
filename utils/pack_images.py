import rpack


def _change_dim_order(sizes):
    return [[s[1], s[0]] for s in sizes]


def get_pack_coords(sizes):
    # list of [height, width] i.e. img.shape order
    sizes = _change_dim_order(sizes)
    positions = rpack.pack(sizes)
    return _change_dim_order(positions)
