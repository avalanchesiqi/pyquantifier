import numpy as np
from matplotlib import pyplot as plt
from collections.abc import Iterable
from sklearn.linear_model import LogisticRegression

from calibration_extrapolation.util import prepare_canvas, get_bin_idx


class CalibrationCurve:
    """
    A calibration curve.
    """
    def __init__(self):
        self.x_axis = None
        self.y_axis = None
    
    def get_calibrated_prob(self, cxs):
        pass

    def plot(self, pos_color='#3d85c6', neg_color='#cc0000', show_diagonal=False, fig_name=False, ax=None):
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
                            facecolor=pos_color, alpha=x, lw=0)
            ax.fill_between([left_point, right_point], 
                            [y, y], 
                            [1, 1], 
                            facecolor=neg_color, alpha=x, lw=0)

        ax.plot(self.x_axis, self.y_axis, 'k-', lw=2)

        if show_diagonal:
            ax.plot([0, 1], [0, 1], 'k--', lw=2)

        ax.set_xlabel('$C(X)$')
        ax.set_ylabel('$P(GT=1|C(X))$')
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylim(ymin=0)

        if fig_name:
            plt.savefig(f'{fig_name}.svg', bbox_inches='tight')


class PerfectCC(CalibrationCurve):
    """
    A perfect calibration curve.
    """
    def __init__(self):
        super().__init__()
        self.x_axis = np.linspace(0, 1, 200)
        self.y_axis = self.x_axis

    def get_calibrated_prob(self, cxs):
        return cxs


class CalibrationLookupTable(CalibrationCurve):
    """
    .get_calibrated_score(cx) finds the nearest cx values above and below it, looks up the LabelDensity for each,
    and makes a new LabelDensity that, for each label,
    interpolates linearly between the values in the two LabelDensity instances.
    """
    def __init__(self, df, num_bin):
        super().__init__()
        self.num_bin = num_bin
        self.x_axis = np.linspace(0, 1, num_bin + 1)

        pos_cx = df[df['GT'] == True]['C(X)'].values
        all_cx = df['C(X)'].values

        pos_hist_freq, _ = np.histogram(pos_cx, bins=self.x_axis)
        all_hist_freq, _ = np.histogram(all_cx, bins=self.x_axis)
        self.y_axis = pos_hist_freq / all_hist_freq

        bin_width = 1 / num_bin
        bin_margin = bin_width / 2
        self.x_axis = self.x_axis[:-1] + bin_margin
    
    def get_calibrated_prob(self, cxs):
        if isinstance(cxs, Iterable):
            return np.array([self.y_axis[get_bin_idx(cx, self.num_bin)] for cx in cxs])
        else:
            return self._find_cali_prob(cxs)


class PlattScaling(CalibrationCurve):
    """
    A logistic calibration curve.
    """
    def __init__(self):
        super().__init__()
        self.lr_regressor = LogisticRegression()

    def fit(self, df):
        train_CX = df['C(X)'].values.reshape(-1, 1)
        train_GT = df['GT'].astype('bool').values
        self.lr_regressor = LogisticRegression(solver='lbfgs', fit_intercept=True).fit(train_CX, train_GT)
        self.x_axis = np.linspace(0, 1, 101)
        self.y_axis = self.get_calibrated_prob(self.x_axis)
        
    def sef_params(self, w, b):
        self.lr_regressor.coef_ = np.array([[w]])
        self.lr_regressor.intercept_ = np.array([b])
        self.lr_regressor.classes_ = np.array([0, 1])
    
    def get_calibrated_prob(self, cxs):
        return self.lr_regressor.predict_proba(cxs.reshape(-1, 1))[:, 1]


class TemperatureScaling(CalibrationCurve):
    """
    A temperature calibration curve.
    """
    def __init__(self):
        super().__init__()
        self.lr_regressor = LogisticRegression()


class InferredCalibrationCurve(CalibrationCurve):
    def __int__(self):
        super().__int__()
