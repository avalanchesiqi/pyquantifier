import numpy as np

from calibration_extrapolation.conf import *
from calibration_extrapolation.util import ContinuousClassDensity, \
    shift_axis, get_bin_idx


class ClassifierDensity(object):
    """
    ClassifierDensity.
    """
    def __init__(self, cxs):
        self.cxs = cxs
        self.size = len(self.cxs)

        self._generate_map_classifier_density()

    def _generate_map_classifier_density(self):
        hist, _ = np.histogram(self.cxs, bins=fine_axis, density=True)
        self.x_axis = shift_axis(fine_axis)
        self.map_classifier_density = hist

    def get_density(self, cx):
        cx_bin_idx = get_bin_idx(cx)
        return (self.map_classifier_density[cx_bin_idx] +
                self.map_classifier_density[cx_bin_idx + 1]) / 2

    def plot(self):
        pass


class BetaDensity(ClassifierDensity):
    def __int__(self):
        super().__int__()


class NonParametricDensity(ClassifierDensity):
    def __int__(self):
        super().__int__()


class InferredClassifierDensity(ClassifierDensity):
    def __int__(self):
        super().__int__()
