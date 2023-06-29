import random

from pyquantifier.distribution.calibration_curve import \
    InferredPlattScaling, ParametricPlattScaling
from pyquantifier.distribution.class_conditional_density import \
    InferredConditionalDensities, ParametricConditionalDensities
from pyquantifier.distribution.classifier_density import \
    InferredClassifierDensity, ParametricClassifierDensity
from pyquantifier.distribution.label_density import \
    InferredLabelDensity, ParametricLabelDensity
from pyquantifier.plot import *
from pyquantifier.util import *


class EmpiricalJointDistribution:
    """
    A joint density between cx and gt.
    """

    def __init__(self, labeled_data):
        self.labeled_data = labeled_data
        self.cxs = [cx for cx, _ in labeled_data]
        self.labels = [gt for _, gt in labeled_data]
        self.classifier_density = InferredClassifierDensity(self.cxs)
        self.label_density = InferredLabelDensity(self.labels)
        self.class_conditional_densities = InferredConditionalDensities(self.labeled_data)
        self.num_class = self.class_conditional_densities.num_class
        self.calibration_curve = InferredPlattScaling()
        self.calibration_curve.fit(self.labeled_data)

    def get_density(self, score, label):
        pass

    def plot_hist(self, **kwds):
        ax = kwds.get('ax', None)
        if ax is None:
            ax = prepare_canvas()
        img_name = kwds.get('img_name', None)

        cp = eval(f'ColorPalette.CC{self.num_class}')

        prev_hist = np.zeros(num_coarse_bin)
        for color_idx, (class_name, cxs) in enumerate(
                self.class_conditional_densities.class_densities.items()):
            hist, _ = np.histogram(cxs, bins=coarse_axis)
            if color_idx == self.num_class - 1:
                one_gradient_plot(ax, shift_axis(coarse_axis), hist + prev_hist,
                                  bottom_axis=prev_hist,
                                  color=cp[color_idx],
                                  edge_color=ColorPalette.unknown_color)
            else:
                one_gradient_plot(ax, shift_axis(coarse_axis), hist + prev_hist,
                                  bottom_axis=prev_hist,
                                  color=cp[color_idx],
                                  edge=False)
            prev_hist += hist

        ax.set_xlabel('$C(X)$')
        ax.set_ylabel('frequency')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim(ymin=0)
        save_to_img(img_name)

    def visualize_all_distribution(self, **kwds):
        fig, axes = plt.subplots(1, 5, figsize=(16, 3))
        class_conditional_density_obj = InferredConditionalDensities(
            self.labeled_data)
        class_conditional_density_obj.plot_hist(ax=axes[0])

        # label density plot
        label_density_obj = InferredLabelDensity(self.labels)
        label_density_obj.plot_bar(ax=axes[1])

        # joint distribution plot
        self.plot_hist(ax=axes[2], **kwds)

        # classifier density plot
        classifier_density_obj = InferredClassifierDensity(self.cxs)
        classifier_density_obj.plot_hist(ax=axes[3])

        # calibration curve plot
        calibration_curve_obj = InferredPlattScaling()
        calibration_curve_obj.fit(self.labeled_data)
        calibration_curve_obj.plot_line(ax=axes[4])

        for ax in axes.ravel():
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='both', which='major')

        plt.tight_layout()


class ParametricJointDensity1:
    """A joint density, constructed by ParametricLabelDensity and ParametricConditionalDensities
    """
    def __init__(self, label_density: ParametricLabelDensity,
                 class_conditional_densities: ParametricConditionalDensities):
        self.label_density = label_density
        self.class_conditional_densities = class_conditional_densities

    def rvs(self, size):
        """Sample random variates of given `size`.

        Parameters
        ----------
        size : int
            Number of random variates

        Returns
        -------
        rvs : ndarray
            Random variates of given `size`
        """
        class_prevalence_dict = self.label_density.class_prevalence_dict
        class_density_dict = self.class_conditional_densities.class_density_dict
        class_names = class_prevalence_dict.keys()

        all_data_points = []
        for class_name in class_names:
            class_prevalence = class_prevalence_dict[class_name]
            num_sample = int(class_prevalence * size)
            class_density = class_density_dict[class_name]
            sampled_cxs = class_density.rvs(num_sample)
            all_data_points.extend([(cx, class_name) for cx in sampled_cxs])
        return all_data_points


class ParametricJointDensity2:
    """A joint density, constructed by ParametricPlattScaling and ParametricClassifierDensity
    """
    def __init__(self, calibration_curve: ParametricPlattScaling,
                 classifier_density: ParametricClassifierDensity):
        self.calibration_curve = calibration_curve
        self.classifier_density = classifier_density

    def rvs(self, size):
        """Sample random variates of given `size`.

        Parameters
        ----------
        size : int
            Number of random variates

        Returns
        -------
        rvs : ndarray
            Random variates of given `size`
        """
        all_cxs = self.classifier_density.rvs(size)
        all_ccxs = self.calibration_curve.get_calibrated_prob(all_cxs)

        all_data_points = []

        for cx, ccx in zip(all_cxs, all_ccxs):
            gt = random.choices([0, 1], [1 - ccx, ccx])[0]
            if gt:
                all_data_points.append((cx, 'Pos'))
            else:
                all_data_points.append((cx, 'Neg'))

        return all_data_points
