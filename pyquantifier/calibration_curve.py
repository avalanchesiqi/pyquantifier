import numpy as np
from matplotlib import pyplot as plt
from collections.abc import Iterable
from sklearn.linear_model import LogisticRegression

from pyquantifier.util import get_bin_idx
from pyquantifier.plot import *


class CalibrationCurve:
    """
    A calibration curve.
    """

    def __init__(self):
        self.x_axis = None
        self.y_axis = None

    def get_calibrated_prob(self, cxs):
        pass

    def plot(self, show_diagonal=False, fig_name=False, ax=None):
        if ax is None:
            ax = prepare_canvas()

        bin_width = 1 / len(self.x_axis)
        bin_margin = bin_width / 2
        for x, y in zip(self.x_axis, self.y_axis):
            left_point = x - bin_margin
            right_point = x + bin_margin

            ax.fill_between([left_point, right_point],
                            [0, 0],
                            [y, y],
                            facecolor=ColorPalette.CC2[1], alpha=x, lw=0)
            ax.fill_between([left_point, right_point],
                            [y, y],
                            [1, 1],
                            facecolor=ColorPalette.CC2[0], alpha=x, lw=0)

        ax.plot(self.x_axis, self.y_axis, 'k-', lw=2)

        if show_diagonal:
            ax.plot([0, 1], [0, 1], 'k--', lw=2)

        ax.set_xlabel('$C(X)$')
        ax.set_ylabel('$P(GT=1|C(X))$')
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylim(ymin=0)

        if fig_name:
            plt.savefig(f'{fig_name}.svg', bbox_inches='tight')


class NonParametricCalibrationCurve(CalibrationCurve):
    def set_x_axis(self, x_axis):
        self.x_axis = x_axis

    def set_y_axis(self, y_axis):
        self.y_axis = y_axis


class PlattScaling(CalibrationCurve):
    """
    A logistic calibration curve. Set parameters directly
    """

    def __init__(self):
        super().__init__()
        self.lr_regressor = LogisticRegression()

    def set_params(self, w, b):
        self.lr_regressor.coef_ = np.array([[w]])
        self.lr_regressor.intercept_ = np.array([b])
        self.lr_regressor.classes_ = np.array([0, 1])
        self.x_axis = np.linspace(0, 1, 101)
        self.y_axis = self.get_calibrated_prob(self.x_axis)

    def get_params(self):
        return self.lr_regressor.coef_, self.lr_regressor.intercept_

    def get_calibrated_prob(self, cxs):
        return self.lr_regressor.predict_proba(cxs.reshape(-1, 1))[:, 1]