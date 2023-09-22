from abc import ABC, abstractmethod
from collections.abc import Iterable
import numpy as np
import matplotlib.pyplot as plt
from pyquantifier.calibration_curve import CalibrationCurve, \
    NonParametricCalibrationCurve
from pyquantifier.plot import *


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

    def plot(self, **kwds):
        plot_empirical_bar({k: v for k, v in zip(self.labels, self.probs)}, **kwds)


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
    def get_density(self):
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
    """A theoretical, parametric mixture continuous univariate distribution class.

    `MixtureCUD` is a base class to generate continuous data samples
    of a random variable from a collection of theoretical parametric
    sub-distributions with weights.

    Parameters
    ----------
    components : list
        Theoretical parametric sub-distributions (mixture components) to
        constitute the mixture distribution

    weights : list
        Non-negative weights for the mixture components

    Methods
    ----------
    get_density
    generate_data
    plot

    Examples
    --------
    If you want to generate a mixture distribution consisting of two Beta
    distributions and one uniform distribution, and the weights for each
    compoment are 2:7:1, you can do the following:

    >>> from scipy.stats import beta, uniform
    >>> md_rv = MixtureCUD(components=[beta(8, 2), beta(2, 5), uniform(0, 1)],
    ...                    weights=[2, 7, 1])

    You can sample 10,000 data points from the mixture distribution,

    >>> md_rv.generate_data(n=10000)

    You can obtain the probability density at any x

    >>> import numpy as np
    >>> md_rv.get_density(np.linspace(0, 1, 101))

    """
    def __init__(self, components, weights):
        self.components = components
        self.num_component = len(components)
        weight_sum = sum(weights)
        self.weights = [weight / weight_sum for weight in weights]

    def pdf(self, score):
        """Probability density at `score`.

        Parameters
        ----------
        x : array_like
            quantiles

        Returns
        -------
        pdf : ndarray
            Probability density evaluated at x
        """
        return sum([weight * component.pdf(score)
                    for weight, component in
                    zip(self.weights, self.components)])

    def get_density(self, scores):
        """Probability density at `scores`.
        """
        if isinstance(scores, Iterable):
            return [self.pdf(score) for score in scores]
        else:
            return self.pdf(scores)

    def generate_data(self, n):
        """Sample random variates of given size `n`.

        Parameters
        ----------
        n : int
            Number of samples to generate

        Returns
        -------
        rvs : ndarray
            Random variates of given `size`
        """
        component_choices = np.random.choice(range(self.num_component),
                                             size=n,
                                             p=self.weights)
        component_samples = [component.rvs(size=n)
                             for component in self.components]
        data_sample = np.choose(component_choices, component_samples)
        return data_sample
    
    def plot(self, **kwds):
        """Plot the probability density function for the mixture distribution.

        Parameters
        ----------
        ax : AxesSubplot, optional
            Axis to plot the pdf
        color : str, optional
            Main color for the plot
        fig_name : str, optional
            Filepath of the output figure
        """
        ax = kwds.pop('ax', None)
        if ax is None:
            ax = prepare_canvas()
        img_name = kwds.pop('img_name', None)
        num_bin = kwds.pop('num_bin', 500)
        weight = kwds.pop('weight', 1)
        return_bottom = kwds.pop('return_bottom', False)

        x_axis = np.linspace(0, 1, num_bin)
        y_axis = np.array([weight * self.get_density(x) for x in x_axis])

        one_gradient_plot(ax, x_axis, y_axis, **kwds)

        ax.set_xlabel('score')
        ax.set_ylabel('density')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim(ymin=0)
        save_to_img(img_name)

        if return_bottom:
            return y_axis

    # def plot(self, ax=None, bottom=None, return_bottom=False, weight=1, num_bin=100, density=False):
    #     if ax is None:
    #         ax = plt.gca()

    #     # plot a line chart of the distribution
    #     x_axis = np.linspace(0, 1, num_bin)
    #     if bottom:
    #         y_axis = [self.get_density(x) * weight + b_y for x, b_y in zip(x_axis, bottom)]
    #     else:
    #         y_axis = [self.get_density(x) * weight for x in x_axis]
    #     if density:
    #         y_axis = np.array(y_axis) / sum(y_axis)
    #     ax.plot(x_axis, y_axis)
    #     ax.set_xlabel('Score')
    #     ax.set_ylabel('Density')
    #     # plt.show()
    #     if return_bottom:
    #         return y_axis


class BinnedCUD(ContinuousUnivariateDistribution):
    def __init__(self, data, bins=10):
        self.data = np.array(data)
        self.bins = bins
    
    def get_length(self):
        return self.bins

    def get_density(self, score):
        hist, _ = np.histogram(self.data, bins=self.bins, density=True)
        return hist[np.searchsorted(np.linspace(0, 1, self.bins), score) - 1]

    def pdf(self, score):
        hist, _ = np.histogram(self.data, bins=self.bins, density=True)
        return hist[np.searchsorted(np.linspace(0, 1, self.bins), score) - 1]

    def pdfs(self):
        hist, _ = np.histogram(self.data, bins=self.bins, density=True)
        return hist

    def sample(self, n):
        # sample without replacement n items from all data
        return np.random.choice(self.data, size=n, replace=False)

    def plot(self, **kwds):
        # plot a line chart of the distribution
        # if 0 < weight <= 1:
        #     new_data = list(self.data) * int(100 * weight)
        # else:
        ax = kwds.pop('ax', None)
        if ax is None:
            ax = prepare_canvas()
        img_name = kwds.pop('img_name', None)
        num_bin = kwds.pop('num_bin', 500)
        weight = kwds.pop('weight', 1)
        return_bottom = kwds.pop('return_bottom', False)
        density = kwds.pop('density', False)
        bottom = kwds.pop('bottom', None)

        new_data = self.data
        if bottom is not None:
            bottom, _, _ = ax.hist(new_data, bins=num_bin, histtype='step',
                                    bottom=bottom, density=density)
        else:
            bottom, _, _ = ax.hist(new_data, bins=num_bin, histtype='step', density=density)
        ax.set_xlabel('score')
        ax.set_ylabel('density')
        if return_bottom:
            return bottom


class NonParametricCUD(ContinuousUnivariateDistribution):
    def __init__(self, x_axis, y_axis):
        self.x_axis = np.array(x_axis)
        self.y_axis = np.array(y_axis)

    def get_density(self, score):
        return score[np.searchsorted(self.x_axis, score) - 1]

    def sample(self, n):
        pass  # TODO

    def plot(self, **kwds):
        ax = kwds.pop('ax', None)
        if ax is None:
            ax = prepare_canvas()
        return_bottom = kwds.pop('return_bottom', False)

        one_gradient_plot(ax, self.x_axis, self.y_axis, **kwds)

        ax.set_xlabel('score')
        ax.set_ylabel('density')

        if return_bottom:
            return self.y_axis


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

    def plot_five_distributions(self, num_bin=500):
        fig, axes = plt.subplots(1, 5, figsize=(16, 3))
        axes = axes.ravel()

        for label in self.labels[::-1]:
            self.class_conditional_densities[label].plot(ax=axes[0], color=eval(f'ColorPalette.{label}_color'),
                                                         num_bin=num_bin, density=True)
        axes[0].set_title('Class Conditional Densities')

        self.label_distribution.plot(ax=axes[1])
        axes[1].set_title('Label Density')

        if isinstance(self, IntrinsicJointDistribution):
            prev_bottom = None
            for label in self.labels:
                weight = self.label_distribution.get_density(label)
                prev_bottom = self.class_conditional_densities[label].plot(
                    ax=axes[2], num_bin=num_bin, color=eval(f'ColorPalette.{label}_color'), 
                    bottom_axis=prev_bottom, return_bottom=True, weight=weight)
        else:
            x_axis = np.linspace(0, 1, num_bin)
            curve_pos = self.calibration_curve.get_calibrated_prob(x_axis) * \
                        np.array([self.classifier_score_distribution.get_density(x)
                                for x in x_axis])
            self.classifier_score_distribution.plot(ax=axes[2], num_bin=num_bin, color=ColorPalette.neg_color, density=True)
            self.class_conditional_densities['pos'].plot(ax=axes[2], weight=self.label_distribution.get_density('pos'), num_bin=num_bin, color=ColorPalette.pos_color, density=True)
        axes[2].set_title('Joint Density')

        self.classifier_score_distribution.plot(ax=axes[3], num_bin=num_bin)
        axes[3].set_title('Classifier Score Density')

        self.calibration_curve.plot(ax=axes[4], show_diagonal=False)
        axes[4].set_title('Calibration Curve')

        for ax in axes:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='both', which='major')
        
        plt.tight_layout()


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

    def calculate_calibration_curve(self, num_bin=1000):
        # calculate the calibration curve
        x_axis = np.linspace(0, 1, num_bin)
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

    def calculate_label_distribution(self, num_bin=1000):
        x_axis = np.linspace(0, 1, num_bin)
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
        return MultinomialDUD(['neg', 'pos'], np.array([area_neg, area_pos]))

    def calculate_class_conditional_densities(self, num_bin=1000):
        x_axis = np.linspace(0, 1, num_bin)
        curve_pos = self.calibration_curve.get_calibrated_prob(x_axis) * \
                    np.array([self.classifier_score_distribution.get_density(x)
                            for x in x_axis])
        curve_neg = (1 - self.calibration_curve.get_calibrated_prob(x_axis)) * \
                    np.array([self.classifier_score_distribution.get_density(x)
                            for x in x_axis])
        curve_pos = np.array(curve_pos) / sum(curve_pos) * num_bin
        curve_neg = np.array(curve_neg) / sum(curve_neg) * num_bin
        return {'pos': NonParametricCUD(x_axis, curve_pos),
                'neg': NonParametricCUD(x_axis, curve_neg)}
