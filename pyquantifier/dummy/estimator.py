import numpy as np
from sklearn.linear_model import LogisticRegression


class CalibrationCurve:
    def __init__(self):
        pass

    def get_calibrated_prob(self, cx):
        pass


class PerfectCalibrationCurve(CalibrationCurve):
    def __init__(self):
        super().__init__()

    def get_calibrated_prob(self, cx):
        return cx


class LogisticCalibrationCurve(CalibrationCurve):
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


class Estimator:
    """
    A dummpy Estimator.
    """
    def __init__(self, classifier_scores):
        self.cxs = classifier_scores
        self.calibration_curve = None

    def set_calibration_curve(self, calibration_curve: CalibrationCurve):
        self.calibration_curve = calibration_curve

    def estimate(self):
        calibrated_probs = np.array([
            self.calibration_curve.get_calibrated_prob(cx) for cx in self.cxs])
        return np.mean(calibrated_probs)
