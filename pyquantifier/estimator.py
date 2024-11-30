import numpy as np

from pyquantifier.distributions import MixtureCUD
from pyquantifier.calibration_curve import CalibrationCurve
from pyquantifier.plot import plot_stacked_frequency

class IntrinsicPrevalenceEstimator:
    def __init__(self):
        pass


class MixtureModelEstimator(IntrinsicPrevalenceEstimator):
    """
    A class for mixture model estimator.
    """

    def __init__(self):
        super().__init__()
        self.positivity_density = None
        self.negativity_density = None

    def set_positive_density(self, positivity_density):
        self.positivity_density = positivity_density

    def set_negativity_density(self, negativity_density):
        self.negativity_density = negativity_density

    # def fit(self, sample_df, base_cx, num_bin=10):
    #     x_axis = np.linspace(0, 1, num_bin + 1)
    #     base_cx_hist, _ = np.histogram(base_cx, bins=x_axis, density=True)
    #     sample_cx_hist, _ = np.histogram(sample_df['C(X)'].values, bins=x_axis, density=True)

    #     weight = base_cx_hist / sample_cx_hist

    #     pos_cx = sample_df[sample_df['GT'] == True]['C(X)'].values
    #     neg_cx = sample_df[sample_df['GT'] == False]['C(X)'].values

    #     pos_hist_freq, _ = np.histogram(pos_cx, bins=x_axis, density=True)
    #     neg_hist_freq, _ = np.histogram(neg_cx, bins=x_axis, density=True)

    #     pos_hist_freq *= weight
    #     pos_total = np.sum(pos_hist_freq)
    #     pos_hist_freq /= pos_total

    #     neg_hist_freq *= weight
    #     neg_total = np.sum(neg_hist_freq)
    #     neg_hist_freq /= neg_total

    #     self.positivity_density = pos_hist_freq
    #     self.negativity_density = neg_hist_freq

    def hellinger(self, p, q):
        return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)

    def estimate(self, cx_array):
        if isinstance(cx_array, list) or isinstance(cx_array, np.ndarray):
            num_bin = self.positivity_density.num_bin
            cx_hist, _ = np.histogram(cx_array, bins=np.linspace(0, 1, num_bin+1), density=True)
        elif isinstance(cx_array, MixtureCUD):
            num_bin = cx_array.num_bin
            cx_hist = cx_array.y_axis

        x_axis = self.positivity_density.x_axis
        min_dist = 10000
        best_p_p = 0

        positive_shape = np.array([self.positivity_density.get_density(x) for x in x_axis])
        negative_shape = np.array([self.negativity_density.get_density(x) for x in x_axis])

        # print(cx_hist)
        # print(positive_shape)
        # print(negative_shape)

        for p_p in np.arange(0, 1.001, 0.001):
            dist = self.hellinger(cx_hist, 
                                  positive_shape * p_p + negative_shape * (1 - p_p))
            # print(f'{p_p=:.3f} {dist=:.3f}')
            if dist < min_dist:
                min_dist = dist
                best_p_p = p_p
        return best_p_p

    def plot(self, cx_array, num_bin=100):
        x_axis = np.linspace(0, 1, num_bin + 1)
        freq_hist, _ = np.histogram(cx_array, bins=x_axis)

        num_bin = len(x_axis)
        bin_width = 1 / num_bin
        bin_margin = bin_width / 2

        x_axis = x_axis[:-1] + bin_margin
        plot_stacked_frequency(x_axis, freq_hist, self.calibration_curve, ax=None, fig_name=None)


class VaryingThresholdEstimator(IntrinsicPrevalenceEstimator):
    """
    A class for varying threshold estimator.
    """

    def __init__(self):
        super().__init__()
        pass


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
        calibrated_prob_array = self.calibration_curve.get_calibrated_probs(cx_array)
        return np.mean(calibrated_prob_array)

    def plot(self, cx_array, num_bin=100):
        x_axis = np.linspace(0, 1, num_bin + 1)
        freq_hist, _ = np.histogram(cx_array, bins=x_axis)

        num_bin = len(x_axis)
        bin_width = 1 / num_bin
        bin_margin = bin_width / 2

        x_axis = x_axis[:-1] + bin_margin
        plot_stacked_frequency(x_axis, freq_hist, self.calibration_curve, ax=None, fig_name=None)