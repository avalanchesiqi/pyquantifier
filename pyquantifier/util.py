import numpy as np


class DiscreteClassDensity:
    """
    discrete class density.
    """
    def __init__(self):
        pass


class ContinuousClassDensity:
    """
    continuous class density.
    """
    def __init__(self):
        pass


def get_bin_idx(score, size=10):
    return min(int(score * size), size-1)


def get_binned_x_axis(num_bin=10):
    return np.arange(0.5/num_bin, 1, 1/num_bin)
