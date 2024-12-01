import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax, logit

from pyquantifier.plot import ColorPalette, prepare_canvas
from pyquantifier.util import get_bin_idx


class CalibrationCurve:
    """
    Implementation of a calibration curve.
    """
    def __init__(self):
        self.num_bin = 100
        self.x_axis = np.arange(0.05/self.num_bin, 1, 1/self.num_bin)
        self.y_axis = self.get_calibrated_probs(self.x_axis)

    def get_calibrated_prob(self, cx):
        pass

    def get_calibrated_probs(self, cxs):
        return np.array([self.get_calibrated_prob(cx) for cx in cxs])

    def plot(self, **kwds):
        ax = kwds.pop('ax', None)
        if ax is None:
            ax = prepare_canvas()

        show_diagonal = kwds.pop('show_diagonal', False)
        filled = kwds.pop('filled', True)
        lc = kwds.pop('lc', 'k')
        label = kwds.pop('label', None)

        bin_width = 1 / self.num_bin
        bin_margin = bin_width / 2

        for bin_idx in range(self.num_bin):
            x = self.x_axis[bin_idx]
            y = self.y_axis[bin_idx]
            left_coord = x - bin_margin
            right_coord = x + bin_margin

            ax.plot([left_coord, right_coord], [y, y], c=lc, lw=2, zorder=40)
            if bin_idx > 0:
                # left edge of the bin
                prev_y = self.y_axis[bin_idx - 1]
                ax.plot([left_coord, left_coord], [prev_y, y], c=lc, lw=2, zorder=40)
            # next_y = self.y_axis[bin_idx + 1] if bin_idx < num_bin - 1 else 1
            # ax.plot([right_coord, right_coord], [y, next_y], c='k', lw=2, zorder=40)

            if filled:
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
        
        if label is not None:
            ax.legend([label], loc='upper left')

        ax.set_xlabel('$C(x)$')
        ax.set_ylabel('$P(y=1|C(x))$')
        ax.set_xlim(xmin=0)
        ax.set_ylim(ymin=0)


class BinnedCalibrationCurve(CalibrationCurve):
    def __init__(self, x_axis, y_axis):
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.num_bin = len(self.x_axis)

    def get_calibrated_prob(self, cx):
        return self.y_axis[get_bin_idx(cx, size=self.num_bin)]


class PiecewiseLinearCalibrationCurve(CalibrationCurve):
    def __init__(self, x_axis, y_axis, bin_inflections):
        self.original_x_axis = x_axis
        self.original_y_axis = y_axis
        self.original_num_bin = len(self.original_x_axis)
        self.original_bin_inflections = bin_inflections
        super().__init__()

    def get_calibrated_prob(self, cx):
        idx = get_bin_idx(cx, size=self.original_num_bin)
        current_bin_inflection = self.original_bin_inflections[idx]

        if cx > current_bin_inflection:
            next_idx = min(idx + 1, self.original_num_bin - 1)
            next_bin_inflection = self.original_bin_inflections[next_idx]
            if next_bin_inflection != current_bin_inflection:
                weight = (cx - current_bin_inflection) / (next_bin_inflection - current_bin_inflection)
                return (1 - weight) * self.original_y_axis[idx] + weight * self.original_y_axis[next_idx]
        elif cx < current_bin_inflection:
            prev_idx = max(idx - 1, 0)
            prev_bin_inflection = self.original_bin_inflections[prev_idx]
            if prev_bin_inflection != current_bin_inflection:
                weight = (cx - prev_bin_inflection) / (current_bin_inflection - prev_bin_inflection)
                return (1 - weight) * self.original_y_axis[prev_idx] + weight * self.original_y_axis[idx]
        # cx == current_bin_inflection 
        # or next_bin_inflection == current_bin_inflection 
        # or prev_bin_inflection == current_bin_inflection
        return self.original_y_axis[idx]


class PlattScaling(CalibrationCurve):
    """
    A logistic calibration curve
    """
    def __init__(self, model):
        self.model = model
        super().__init__()
    
    def get_calibrated_prob(self, cx):
        return self.model.predict_proba(cx.reshape(1, -1))[0, 1]

    def get_calibrated_probs(self, cxs):
        return self.model.predict_proba(cxs.reshape(-1, 1))[:, 1]


class IsotonicRegressionCalibrationCurve(CalibrationCurve):
    def __init__(self, model):
        self.model = model
        super().__init__()

    def get_calibrated_prob(self, cx):
        return self.model.predict(np.array(cx))[0]
    
    def get_calibrated_probs(self, cxs):
        return self.model.predict(np.array(cxs))


class TemperatureScaling(CalibrationCurve):
    """
    A logistic calibration curve
    """
    def __init__(self, temperature):
        self.temperature = temperature
        super().__init__()
        
    # Define the loss function (negative log-likelihood)
    @staticmethod
    def nll_loss(temperature, logits, labels):
        scaled_logits = logits / temperature
        if scaled_logits.ndim == 1:
            probs = 1 / (1 + np.exp(-scaled_logits))  # Sigmoid for binary classification
            nll = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
        else:
            probs = softmax(scaled_logits, axis=1)  # Softmax for multi-class classification
            nll = -np.mean(np.log(probs[np.arange(len(labels)), labels]))
        return nll

    def fit(self, X, labels):
        # Ensure logits and labels are numpy arrays
        X = np.hstack((1 - X, X))
        logits = np.array(logit(X))
        labels = np.array(labels)

        # Optimize the temperature parameter
        result = minimize(self.nll_loss, self.temperature, bounds=[(0.1, 10)], args=(logits, labels))
        self.temperature = result.x[0]
        return self.nll_loss(self.temperature, logits, labels)

    def transform(self, logits):
        # Apply temperature scaling
        return logits / self.temperature

    def get_calibrated_probs(self, X):
        # Apply temperature scaling and softmax or sigmoid
        # concatenate x_axis and x_axis2 into a 100 x 2 array
        X = np.hstack((1 - X, X))
        logits = np.array(logit(X))
        scaled_logits = self.transform(logits)
        if scaled_logits.ndim == 1:
            return 1 / (1 + np.exp(-scaled_logits))[:, 1]  # Sigmoid for binary classification
        else:
            return softmax(scaled_logits, axis=1)[:, 1]  # Softmax for multi-class classification
