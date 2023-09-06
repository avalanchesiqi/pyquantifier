from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.stats import beta, uniform
import matplotlib.pyplot as plt
from pyquantifier.calibration_curve import CalibrationCurve, \
    NonParametricCalibrationCurve


class UnivariateDistribution(ABC):
    @abstractmethod
    def get_density(self, val):
        pass

    @abstractmethod
    def plot(self):
        pass


class DiscreteUnivariateDistribution(UnivariateDistribution, ABC):
    def __int__(self):
        self.labels = None
        self.probs = None

    def get_density(self, label: str):
        # given a label class, return the density of this class
        return self.probs[self.labels.index(label)]


class MultinomialDUD(DiscreteUnivariateDistribution):
    def __init__(self, labels, probs):
        self.labels = labels
        self.probs = np.array(probs) / np.sum(probs)  # normalize the probs

    def generate_data(self, n):
        # generate a simulated dataset of size n
        return np.random.choice(self.labels, size=n, replace=True, p=self.probs)

    def plot(self, ax=None):
        # plot a bar chart of the distribution
        if ax is None:
            ax = plt.gca()
        ax.bar(self.labels, self.probs)
        ax.set_ylabel('Probability')
        # plt.show()


class BinnedDUD(DiscreteUnivariateDistribution):
    def __init__(self, data):
        self.labels = list(set(data))
        self.data = np.array(data)
        self.probs = np.array([np.mean(self.data == label) for label in
                               self.labels])

    def sample(self, n):
        # sample without replacement n items from all data
        return np.random.choice(self.data, size=n, replace=False)

    def get_ci(self, label):
        # given a label class, return the confidence interval of this class
        p_label = self.get_density(label)
        return 1.96 * np.sqrt(p_label * (1 - p_label) / len(self.data))

    def plot(self, ci=True, ax=None):
        if ax is None:
            ax = plt.gca()
        # plot a bar chart with confidence intervals of the distribution
        if ci:
            ax.bar(self.labels, self.probs, yerr=[self.get_ci(label) for
                                                  label in self.labels])
        else:
            ax.bar(self.labels, self.probs)
        ax.set_ylabel('Probability')
        # plt.show()


class ContinuousUnivariateDistribution(UnivariateDistribution, ABC):
    def get_density(self, label: str):
        pass


# class BetaCUD(ContinuousUnivariateDistribution):
#     def __init__(self, a, b):
#         self.dist = beta(a, b)
#
#     def get_density(self, score):
#         return self.dist.pdf(score)
#
#     def generate_data(self, n):
#         # generate a simulated dataset of size n
#         return self.dist.rvs(n)
#
#     def plot(self):
#         # plot a bar chart of the distribution
#         x_axis = np.linspace(0, 1, 100)
#         y_axis = [self.get_density(x) for x in x_axis]
#         plt.plot(x_axis, y_axis)
#         plt.xlabel('Score')
#         plt.ylabel('Density')
#         plt.show()


class MixtureCUD(ContinuousUnivariateDistribution):
    def __init__(self, components, weights):
        self.components = components
        self.num_component = len(components)
        weight_sum = sum(weights)
        self.weights = [weight / weight_sum for weight in weights]

    def get_density(self, score):
        if isinstance(self.components[0], MixtureCUD):
            return sum([weight * component.get_density(score)
                        for weight, component in
                        zip(self.weights, self.components)])
        else:
            return sum([weight * component.pdf(score)
                        for weight, component in
                        zip(self.weights, self.components)])

    def generate_data(self, n):
        component_choices = np.random.choice(range(self.num_component),
                                             size=n,
                                             p=self.weights)
        component_samples = [component.rvs(size=n)
                             for component in self.components]
        data_sample = np.choose(component_choices, component_samples)
        return data_sample

    def plot(self, ax=None, bottom=None, return_bottom=False):
        if ax is None:
            ax = plt.gca()

        # plot a line chart of the distribution
        x_axis = np.linspace(0, 1, 1000)
        if bottom:
            y_axis = [self.get_density(x) + b_y for x, b_y in zip(x_axis, bottom)]
        else:
            y_axis = [self.get_density(x) for x in x_axis]
        ax.plot(x_axis, y_axis)
        ax.set_xlabel('Score')
        ax.set_ylabel('Density')
        # plt.show()
        if return_bottom:
            return y_axis


class BinnedCUD(ContinuousUnivariateDistribution):
    def __init__(self, data):
        self.data = np.array(data)

    def get_density(self, score, num_bin=100):
        hist, _ = np.histogram(self.data, bins=num_bin, density=True)
        return hist[np.searchsorted(np.linspace(0, 1, num_bin), score) - 1]

    def sample(self, n):
        # sample without replacement n items from all data
        return np.random.choice(self.data, size=n, replace=False)

    def plot(self, num_bin=100, ax=None, bottom=None, return_bottom=False):
        if ax is None:
            ax = plt.gca()
        # plot a line chart of the distribution
        if bottom is not None:
            bottom_hist, _ = ax.hist(self.data, bins=num_bin, density=True,
                                    bottom=bottom)
        else:
            ax.hist(self.data, bins=num_bin, density=True)
        ax.set_xlabel('Score')
        ax.set_ylabel('Density')
        # plt.show()
        if return_bottom:
            return bottom_hist


class NonParametricCUD(ContinuousUnivariateDistribution):
    def __init__(self, x_axis, y_axis):
        self.x_axis = np.array(x_axis)
        self.y_axis = np.array(y_axis)

    def get_density(self, score):
        return score[np.searchsorted(self.x_axis, score) - 1]

    def sample(self, n):
        pass  # TODO

    def plot(self, num_bin=100, ax=None, bottom=None, return_bottom=False):
        if ax is None:
            ax = plt.gca()
        # plot a line chart of the distribution
        if bottom is not None:
            y_axis = self.y_axis + bottom
        else:
            y_axis = self.y_axis
        ax.plot(self.x_axis, y_axis)
        ax.set_xlabel('Score')
        ax.set_ylabel('Density')
        if return_bottom:
            return y_axis


class JointDistribution:
    def __init__(self, labels):
        self.labels = labels
        self.label_distribution = None
        self.class_conditional_densities = None
        self.classifier_score_distribution = None
        self.calibration_curve = None

    def sample(self, n):
        pass

    def get_density(self, score, label):
        pass

    def plot_five_distributions(self):
        fig, axes = plt.subplots(1, 5, figsize=(16, 3))
        axes = axes.ravel()

        for label in self.labels:
            self.class_conditional_densities[label].plot(ax=axes[0])
        axes[0].set_title('Classifier Conditional Densities')

        self.label_distribution.plot(ax=axes[1])
        axes[1].set_title('Label Distribution')

        prev_bottom = None
        for label in self.labels:
            prev_bottom = self.class_conditional_densities[label].plot(
                ax=axes[2], bottom=prev_bottom, return_bottom=True)
        axes[2].set_title('Joint Distribution')

        self.classifier_score_distribution.plot(ax=axes[3])
        axes[3].set_title('Classifier Score Distribution')

        self.calibration_curve.plot(ax=axes[4], show_diagonal=False)
        axes[4].set_title('Calibration Curve')

        for ax in axes:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='both', which='major')


class IntrinsicJointDistribution(JointDistribution):
    def __init__(self, labels: list,
                 label_distribution: DiscreteUnivariateDistribution,
                 class_conditional_densities: dict):
        super().__init__(labels)
        self.label_distribution = label_distribution
        self.class_conditional_densities = class_conditional_densities
        self.classifier_score_distribution = self.calculate_classifier_score_distribution()
        self.calibration_curve = self.calculate_calibration_curve()

    def calculate_classifier_score_distribution(self):
        # calculate the classifier score distribution
        # by marginalizing out the label distribution
        components = [self.class_conditional_densities[label]
                      for label in self.labels]
        weights = [self.label_distribution.get_density(label)
                   for label in self.labels]
        return MixtureCUD(components, weights)

    def calculate_calibration_curve(self):
        # calculate the calibration curve
        x_axis = np.linspace(0, 1, 1000)
        y_axis = [self.class_conditional_densities['pos'].get_density(x) *
                  self.label_distribution.get_density('pos') /
                  self.classifier_score_distribution.get_density(x)
                  for x in x_axis]
        ret = NonParametricCalibrationCurve()
        ret.set_x_axis(x_axis)
        ret.set_y_axis(y_axis)
        return ret


class ExtrinsicJointDistribution(JointDistribution):
    def __init__(self, labels: list,
                 classifier_score_distribution: ContinuousUnivariateDistribution,
                 calibration_curve: CalibrationCurve):
        super().__init__(labels)
        self.classifier_score_distribution = classifier_score_distribution
        self.calibration_curve = calibration_curve
        self.label_distribution = self.calculate_label_distribution()
        self.class_conditional_densities = self.calculate_class_conditional_densities()

    def calculate_label_distribution(self):
        x_axis = np.linspace(0, 1, 1000)
        curve_pos = self.calibration_curve.get_calibrated_prob(x_axis) * \
                    np.array([self.classifier_score_distribution.get_density(x)
                            for x in x_axis])
        curve_neg = (1 - self.calibration_curve.get_calibrated_prob(x_axis)) * \
                    np.array([self.classifier_score_distribution.get_density(x)
                            for x in x_axis])
        area_pos = sum(curve_pos)
        area_neg = sum(curve_neg)
        total_area = area_pos + area_neg
        area_pos /= total_area
        area_neg /= total_area
        return MultinomialDUD(self.labels, np.array([area_pos, area_neg]))

    def calculate_class_conditional_densities(self):
        x_axis = np.linspace(0, 1, 1000)
        curve_pos = self.calibration_curve.get_calibrated_prob(x_axis) * \
                    np.array([self.classifier_score_distribution.get_density(x)
                            for x in x_axis])
        curve_neg = (1 - self.calibration_curve.get_calibrated_prob(x_axis)) * \
                    np.array([self.classifier_score_distribution.get_density(x)
                            for x in x_axis])
        curve_pos = np.array(curve_pos) / sum(curve_pos)
        curve_neg = np.array(curve_neg) / sum(curve_neg)
        return {'pos': NonParametricCUD(x_axis, curve_pos),
                'neg': NonParametricCUD(x_axis, curve_neg)}
