import os
from matplotlib import pyplot as plt
from pyquantifier.conf import *


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



