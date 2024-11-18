from abc import abstractmethod
import numpy as np
from collections.abc import Iterable
from sklearn.linear_model import LogisticRegression

from pyquantifier.plot import *
from pyquantifier.util import get_bin_idx


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
        one_gradient_plot(ax, self.x_axis, self.y_axis, color=ColorPalette.pos, edge_color='k')
        one_gradient_plot(ax, self.x_axis, top_axis=np.ones(num_bin), bottom_axis=self.y_axis, color=ColorPalette.neg, edge=False)

        ax.plot(self.x_axis, self.y_axis, 'k-', lw=2)

        if show_diagonal:
            ax.plot([0, 1], [0, 1], 'k--', lw=2)

        ax.set_xlabel('$C(x)$')
        ax.set_ylabel('$P(y=1|C(x))$')
        ax.set_xlim(xmin=0)
        ax.set_ylim(ymin=0)


class BinnedCalibrationCurve(CalibrationCurve):
    def __init__(self, x_axis, y_axis):
        self.x_axis = x_axis
        self.y_axis = y_axis

    def set_x_axis(self, x_axis):
        self.x_axis = x_axis

    def set_y_axis(self, y_axis):
        self.y_axis = y_axis

    def get_calibrated_prob(self, cxs):
        # print(len(self.x_axis), len(self.y_axis))
        # for cx in cxs:
        #     if np.searchsorted(self.x_axis, cx) > 9:
        #         print(self.x_axis)
        #         print(cx, np.searchsorted(self.x_axis, cx), get_bin_idx(cx, len(self.x_axis)))
        # return np.array([self.y_axis[np.searchsorted(self.x_axis, score)] for score in cxs])
        return np.array([self.y_axis[get_bin_idx(cx, len(self.x_axis))] for cx in cxs])
    
    def get_xy_values_for_step_function(self):        
        # Create step function for x_axis and y_axis
        # Make two x-axis points for each bin, giving the left and right edges, and repeat the y-axis value for each bin.
        # Assume that the bins are equal width, so we can calculate the bin edges using linspace.
        step_x = np.repeat(np.linspace(0, 1, len(self.x_axis) + 1), 2)[1:-1]
        step_y = np.repeat(self.y_axis, 2)
        
        return step_x, step_y

    
    def plot(self, **kwds):
        # plot as a step function
        ax = kwds.pop('ax', None)
        if ax is None:
            ax = prepare_canvas()
    
        show_diagonal = kwds.pop('show_diagonal', False)
        alpha = kwds.pop('alpha', 1)
    
        step_x, step_y = self.get_xy_values_for_step_function()
    
        # Fill the area below the step function
        ax.fill_between(step_x,
                        np.zeros(len(step_x)),
                        step_y,
                        facecolor=ColorPalette.pos, alpha=alpha, lw=0)
    
        # Fill the area above the step function
        ax.fill_between(step_x,
                        step_y,
                        np.ones(len(step_x)),
                        facecolor=ColorPalette.neg, alpha=alpha, lw=0)
    
        # # Plot the step function
        # ax.step(step_x, step_y, where='post', color='k', lw=2)
    
        if show_diagonal:
            ax.plot([0, 1], [0, 1], 'k--', lw=1)
    
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Curve')

class PiecewiseLinearCalibrationCurve(BinnedCalibrationCurve):
    def __init__(self, x_axis, y_axis, bin_means):
        super().__init__(x_axis, y_axis)
        self.bin_means = bin_means

    def get_xy_values_for_step_function(self):
        # Calculate the left and right sides of each bin, assuming that only the number of bins matters, not the actual x_axis values
        # Unlike in parent class, we want to connect (x,y) points with lines, with bin_means as the x-values. Plus 0 and 1 as x_axis values.

        step_x = np.concatenate([[0], self.bin_means, [1]])
        step_y = np.concatenate([[self.y_axis[0]], self.y_axis, [self.y_axis[-1]]])
        
        return step_x, step_y

    def get_calibrated_prob(self, cxs):
        def get_yval(cx):
            idx = get_bin_idx(cx, len(self.x_axis))
            current_bin_mean = self.bin_means[idx]

            if cx > current_bin_mean:
                next_idx = min(idx + 1, len(self.x_axis) - 1)
                next_bin_mean = self.bin_means[next_idx]
                if next_bin_mean == current_bin_mean:
                    return self.y_axis[idx]
                else:
                    weight = (cx - current_bin_mean) / (next_bin_mean - current_bin_mean)
                    return (1 - weight) * self.y_axis[idx] + weight * self.y_axis[next_idx]
            elif cx < current_bin_mean:
                prev_idx = max(idx - 1, 0)
                prev_bin_mean = self.bin_means[prev_idx]
                if prev_bin_mean == current_bin_mean:
                    return self.y_axis[idx]
                else:
                    weight = (cx - prev_bin_mean) / (current_bin_mean - prev_bin_mean)
                    return (1 - weight) * self.y_axis[prev_idx] + weight * self.y_axis[idx]
            else: # cx == current_bin_mean
                return self.y_axis[idx]

        return np.array([get_yval(cx) for cx in cxs])


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
        self.x_axis = np.arange(0.5/100, 1, 1/100)
        self.y_axis = self.get_calibrated_prob(self.x_axis)

    def get_params(self):
        return self.lr_regressor.coef_, self.lr_regressor.intercept_

    def get_calibrated_prob(self, cxs):
        return self.lr_regressor.predict_proba(cxs.reshape(-1, 1))[:, 1]