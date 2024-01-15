from abc import abstractmethod
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

    @abstractmethod
    def get_calibrated_prob(self):
        pass

    def plot(self, **kwds):
        ax = kwds.pop('ax', None)
        if ax is None:
            ax = prepare_canvas()

        show_diagonal = kwds.pop('show_diagonal', False)
        alpha = kwds.pop('alpha', 1)

        num_bin = len(self.x_axis)
        bin_width = 1 / num_bin
        bin_margin = bin_width / 2

        ax.fill_between(self.x_axis,
                        np.zeros(num_bin),
                        self.y_axis,
                        facecolor=ColorPalette['pos'], alpha=alpha, lw=0)
        ax.fill_between(self.x_axis,
                        self.y_axis,
                        np.ones(num_bin),
                        facecolor=ColorPalette['neg'], alpha=alpha, lw=0)

        # for x, y in zip(self.x_axis, self.y_axis):
        #     left_coord = x - bin_margin
        #     right_coord = x + bin_margin

        #     ax.fill_between([left_coord, right_coord],
        #                     [0, 0],
        #                     [y, y],
        #                     facecolor=ColorPalette['pos'], alpha=x, lw=0)
        #     ax.fill_between([left_coord, right_coord],
        #                     [y, y],
        #                     [1, 1],
        #                     facecolor=ColorPalette['neg'], alpha=x, lw=0)

        ax.plot(self.x_axis, self.y_axis, 'k-', lw=2)

        if show_diagonal:
            ax.plot([0, 1], [0, 1], 'k--', lw=2)

        ax.set_xlabel('$C(x)$')
        ax.set_ylabel('$P(y=1|C(x))$')
        # ax.set_yticks([0, 0.5, 1])
        ax.set_xlim(xmin=0)
        ax.set_ylim(ymin=0)
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_xlabel('')
        # ax.set_ylabel('')


class BinnedCalibrationCurve(CalibrationCurve):
    def __init__(self, x_axis, y_axis):
        self.x_axis = x_axis
        self.y_axis = y_axis

    def set_x_axis(self, x_axis):
        self.x_axis = x_axis

    def set_y_axis(self, y_axis):
        self.y_axis = y_axis

    def get_calibrated_prob(self, cxs):
        return np.array([self.y_axis[np.searchsorted(self.x_axis, score)] for score in cxs])


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