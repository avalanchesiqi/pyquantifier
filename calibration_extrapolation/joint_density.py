from calibration_extrapolation.calibration_curve import CalibrationCurve
from calibration_extrapolation.class_conditional_density import ClassConditionalDensities
from calibration_extrapolation.util import DiscreteClassDensity, ContinuousClassDensity


class JointDensity:
    """
    A joint density between cx and gt.
    """

    def __init__(self,
                 label_density: DiscreteClassDensity = None,
                 conditional_class_densities: ClassConditionalDensities = None,
                 cx_density: ContinuousClassDensity = None,
                 calibration_curve: CalibrationCurve = None):
        if label_density and conditional_class_densities:
            self.label_density = label_density
            self.conditional_class_densities = conditional_class_densities
        elif cx_density and calibration_curve:
            self.cx_density = cx_density
            self.calibration_curve = calibration_curve
        else:
            print('You must initialize the JointDensity class with either set of '
                  'label_density + conditional_class_densities or cx_density + calibration_curve')

    def get_density(self, score, label):
        pass

    def plot(self):
        pass
