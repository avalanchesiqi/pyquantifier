import numpy as np

from calibration_extrapolation.util import plot_stacked_frequency


class IntrinsicEstimator:
    def __init__(self):
        pass


class MixtureModelEstimator(IntrinsicEstimator):
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

    def fit(self, sample_df, base_cx, num_bin=10):
        x_axis = np.linspace(0, 1, num_bin + 1)
        base_cx_hist, _ = np.histogram(base_cx, bins=x_axis, density=True)
        sample_cx_hist, _ = np.histogram(sample_df['C(X)'].values, bins=x_axis, density=True)

        weight = base_cx_hist / sample_cx_hist

        pos_cx = sample_df[sample_df['GT'] == True]['C(X)'].values
        neg_cx = sample_df[sample_df['GT'] == False]['C(X)'].values

        pos_hist_freq, _ = np.histogram(pos_cx, bins=x_axis, density=True)
        neg_hist_freq, _ = np.histogram(neg_cx, bins=x_axis, density=True)

        pos_hist_freq *= weight
        pos_total = np.sum(pos_hist_freq)
        pos_hist_freq /= pos_total

        neg_hist_freq *= weight
        neg_total = np.sum(neg_hist_freq)
        neg_hist_freq /= neg_total

        self.positivity_density = pos_hist_freq
        self.negativity_density = neg_hist_freq

    def hellinger(self, p, q):
        return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)

    def estimate(self, cx_array):
        num_bin = len(self.positivity_density)
        x_axis = np.linspace(0, 1, num_bin + 1)
        cx_hist, _ = np.histogram(cx_array, bins=x_axis, density=True)

        min_dist = 10000
        best_p_p = 0

        for p_p in np.linspace(0, 1, 101):
            dist = self.hellinger(cx_hist, self.positivity_density * p_p + self.negativity_density * (1 - p_p))
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