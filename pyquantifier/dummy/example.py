import numpy as np
from estimator import Estimator, LogisticCalibrationCurve

test_cxs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

hate_cc = LogisticCalibrationCurve()
hate_cc.set_params(w=0.5, b=1)

hate_estimator = Estimator(test_cxs)
hate_estimator.set_calibration_curve(hate_cc)
print(hate_estimator.estimate())
