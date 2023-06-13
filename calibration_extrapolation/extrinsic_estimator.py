import numpy as np

from calibration_extrapolation.calibration_curve import CalibrationCurve
from calibration_extrapolation.util import plot_stacked_frequency


class ExtrinsicPrevalenceEstimator:
    def __init__(self):
        pass


class ProbabilityEstimator(ExtrinsicPrevalenceEstimator):
    """
    A class for probability estimator.
    """

    def __init__(self):
        super().__init__()
        self.calibration_curve = None

    def set_calibration_curve(self, calibration_curve: CalibrationCurve):
        self.calibration_curve = calibration_curve

    def estimate(self, cx_array):
        calibrated_prob_array = self.calibration_curve.get_calibrated_prob(cx_array)
        return np.mean(calibrated_prob_array)

    def plot(self, cx_array, num_bin=100):
        x_axis = np.linspace(0, 1, num_bin + 1)
        freq_hist, _ = np.histogram(cx_array, bins=x_axis)

        num_bin = len(x_axis)
        bin_width = 1 / num_bin
        bin_margin = bin_width / 2

        x_axis = x_axis[:-1] + bin_margin
        plot_stacked_frequency(x_axis, freq_hist, self.calibration_curve, ax=None, fig_name=None)

