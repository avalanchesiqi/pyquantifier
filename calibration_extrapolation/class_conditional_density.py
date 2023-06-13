from calibration_extrapolation.label_density import LabelDensity

class ClassConditionalDensities:
    """
    Class conditional densities.
    """
    def __init__(self):
        pass

    def get_density(self, cx, label):
        pass

    def get_label_density_function(self, label):
        return LabelDensity(label)


class DictionaryOfConditionalDensities(ClassConditionalDensities):
    def __int__(self):
        super().__int__()


class InferredConditionalDensity(ClassConditionalDensities):
    def __int__(self):
        super().__int__()
