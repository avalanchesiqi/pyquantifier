# -*- coding: utf-8 -*-

from calibration_extrapolation.label_density import LabelDensity


class ClassConditionalDensities(object):
    """
    Class conditional densities.
    """
    def __init__(self, label, pdf):
        self.label = label
        self.pdf = pdf

    def get_density(self, cx, label):
        pass

    def get_label_density_function(self, label):
        return LabelDensity(label)

    def plot(self):
        pass


class DictionaryOfConditionalDensities(ClassConditionalDensities):
    def __int__(self):
        super().__int__()


class InferredConditionalDensity(ClassConditionalDensities):
    def __int__(self):
        super().__int__()
