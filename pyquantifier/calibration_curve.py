import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

from pyquantifier.plot import ColorPalette, prepare_canvas
from pyquantifier.util import get_bin_idx


class CalibrationCurve:
    """
    A calibration curve.
    """
    def __init__(self):
        self.num_bin = 100
        self.x_axis = np.arange(0.05/self.num_bin, 1, 1/self.num_bin)
        self.y_axis = self.get_calibrated_prob(self.x_axis)

    def get_calibrated_prob(self, cxs):
        pass
        # def get_calibrated_value(score):
        #     # find the nearest x_axis value below score (or the first x_axis value)
        #     indx = np.searchsorted(self.x_axis, score, side='right') - 1
        #     # Ensure indx is within the valid range
        #     if indx < 0:
        #         indx = 0
        #     return self.y_axis[indx]

        # return np.array([get_calibrated_value(score) for score in cxs])

    def plot(self, **kwds):
        ax = kwds.pop('ax', None)
        if ax is None:
            ax = prepare_canvas()

        show_diagonal = kwds.pop('show_diagonal', False)

        num_bin = len(self.x_axis)
        bin_width = 1 / num_bin
        bin_margin = bin_width / 2

        for bin_idx in range(num_bin):
            x = self.x_axis[bin_idx]
            y = self.y_axis[bin_idx]
            left_coord = x - bin_margin
            right_coord = x + bin_margin

            ax.plot([left_coord, right_coord], [y, y], c='k', lw=2, zorder=40)
            if bin_idx > 0:
                prev_y = self.y_axis[bin_idx - 1]
                ax.plot([left_coord, left_coord], [prev_y, y], c='k', lw=2, zorder=40)
            # next_y = self.y_axis[bin_idx + 1] if bin_idx < num_bin - 1 else 1
            # ax.plot([right_coord, right_coord], [y, next_y], c='k', lw=2, zorder=40)

            ax.fill_between([left_coord, right_coord],
                            [y, y],
                            [0, 0],
                            facecolor=ColorPalette.pos, alpha=x, lw=0)

            ax.fill_between([left_coord, right_coord],
                            [1, 1],
                            [y, y],
                            facecolor=ColorPalette.neg, alpha=x, lw=0)

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
        self.num_bin = len(self.x_axis)

    def get_calibrated_prob(self, cxs):
        # print(len(self.x_axis), len(self.y_axis))
        # for cx in cxs:
        #     if np.searchsorted(self.x_axis, cx) > 9:
        #         print(self.x_axis)
        #         print(cx, np.searchsorted(self.x_axis, cx), get_bin_idx(cx, len(self.x_axis)))
        # return np.array([self.y_axis[np.searchsorted(self.x_axis, score)] for score in cxs])
        return np.array([self.y_axis[get_bin_idx(cx, size=self.num_bin)] for cx in cxs])


class PiecewiseLinearCalibrationCurve(CalibrationCurve):
    def __init__(self, x_axis, y_axis, bin_means):
        self.original_x_axis = x_axis
        self.original_y_axis = y_axis
        self.original_num_bin = len(self.original_x_axis)
        self.original_bin_means = bin_means
        super().__init__()

    def _get_yval(self, cx):
        idx = get_bin_idx(cx, size=self.original_num_bin)
        current_bin_mean = self.original_bin_means[idx]

        if cx > current_bin_mean:
            next_idx = min(idx + 1, self.original_num_bin - 1)
            next_bin_mean = self.original_bin_means[next_idx]
            if next_bin_mean == current_bin_mean:
                return self.original_y_axis[idx]
            else:
                weight = (cx - current_bin_mean) / (next_bin_mean - current_bin_mean)
                return (1 - weight) * self.original_y_axis[idx] + weight * self.original_y_axis[next_idx]
        elif cx < current_bin_mean:
            prev_idx = max(idx - 1, 0)
            prev_bin_mean = self.original_bin_means[prev_idx]
            if prev_bin_mean == current_bin_mean:
                return self.original_y_axis[idx]
            else:
                weight = (cx - prev_bin_mean) / (current_bin_mean - prev_bin_mean)
                return (1 - weight) * self.original_y_axis[prev_idx] + weight * self.original_y_axis[idx]
        else: # cx == current_bin_mean
            return self.original_y_axis[idx]

    def get_calibrated_prob(self, cxs):
        return np.array([self._get_yval(cx) for cx in cxs])


class PlattScaling(CalibrationCurve):
    """
    A logistic calibration curve
    """
    def __init__(self, model):
        self.model = model
        super().__init__()

    def get_calibrated_prob(self, cxs):
        # print(self.model.predict_proba(cxs.reshape(-1, 1))[:, 1])
        return self.model.predict_proba(cxs.reshape(-1, 1))[:, 1]


class TemperatureScaling(CalibrationCurve):
    """
    A logistic calibration curve
    """
    def __init__(self, temperature):
        self.temperature = temperature
        super().__init__()
    
    def get_temperature(self):
        return self.temperature
    
    def _get_logits(self, X):
        X = X + 1e-10
        X /= np.sum(X, axis=-1, keepdims=True)
        return np.log(X)
    
    def fit(self, X, labels):
        # Ensure logits and labels are numpy arrays
        logits = np.array(self._get_logits(X))
        labels = np.array(labels)

        # Define the loss function (negative log-likelihood)
        def nll_loss(temperature):
            scaled_logits = logits / temperature
            if scaled_logits.ndim == 1:
                probs = 1 / (1 + np.exp(-scaled_logits))  # Sigmoid for binary classification
                nll = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
            else:
                probs = softmax(scaled_logits, axis=1)  # Softmax for multi-class classification
                nll = -np.mean(np.log(probs[np.arange(len(labels)), labels]))
            return nll

        # Optimize the temperature parameter
        result = minimize(nll_loss, self.temperature, bounds=[(0.1, 10)])
        self.temperature = result.x[0]

    def transform(self, logits):
        # Apply temperature scaling
        return logits / self.temperature

    def get_calibrated_prob(self, X):
        # Apply temperature scaling and softmax or sigmoid
        X_neg = 1 - X

        # concatenate x_axis and x_axis2 into a 100 x 2 array
        X = np.vstack((X, X_neg)).T
        logits = np.array(self._get_logits(X))
        scaled_logits = self.transform(logits)
        if scaled_logits.ndim == 1:
            return 1 / (1 + np.exp(-scaled_logits))[:, 0]  # Sigmoid for binary classification
        else:
            return softmax(scaled_logits, axis=1)[:, 0]  # Softmax for multi-class classification


class IsotonicRegressionCalibrationCurve(CalibrationCurve):
    def __init__(self, model):
        self.model = model
        super().__init__()

    def get_calibrated_prob(self, cxs):
        return self.model.predict(cxs)
