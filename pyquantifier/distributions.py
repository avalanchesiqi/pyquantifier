import uproot  # go up to the project root

from abc import ABC, abstractmethod
from collections.abc import Iterable
import numpy as np
import matplotlib.pyplot as plt

from pyquantifier.calibration_curve import CalibrationCurve, BinnedCalibrationCurve
from pyquantifier.plot import *


class UnivariateDistribution(ABC):
    """A base class for univariate distributions.
    """
    @abstractmethod
    def get_density(self):
        pass

    @abstractmethod
    def plot(self):
        pass


class EmpiricalData:
    """A base class for empirical data.
    """
    def __int__(self):
        """Initialize the class.
        
        Parameters
        ----------
        data : list
            Data samples
        """
        self.data = None

    def sample(self, n, replace=False):
        """Sample random variates of given size `n`.
        
        Parameters
        ----------
        n : int
            Number of samples to generate
        replace : bool, optional
            Whether to sample with replacement or not
            
        Returns
        -------
        rvs : ndarray
            Random variates of given `size`
        """
        return np.random.choice(self.data, size=n, replace=replace)


class DiscreteUnivariateDistribution(UnivariateDistribution):
    """A base class for discrete univariate distributions.
    """
    def __int__(self):
        """Initialize the class.
        
        Parameters
        ----------
        labels : list
            Class labels
        
        probs : list
            Non-negative probabilities for the labels
        """
        self.labels = None
        self.probs = None
    
    def get_label_prob_dict(self):
        """Get the dictionary of labels and their probabilities.
        
        Returns
        -------
        label_prob_dict : dict
            Dictionary of labels and their probabilities
        """
        return dict(zip(self.labels, self.probs))

    def get_density(self, label: str):
        """Get the probability density of a given label.

        Parameters
        ----------
        label : str
            Label to be evaluated

        Returns
        -------
        density : float
            Probability density of the given label
        """
        return self.label_prob_dict[label]
    
    def get_ci(self):
        pass
    
    def plot(self, **kwds):
        """Plot bar chart of discrete univariate distributions.
        """
        # ax = kwds.pop('ax', None)
        # if ax is None:
        #     ax = prepare_canvas()

        # x_axis = np.arange(len(self.labels))
        # label_axis = sorted(self.labels)
        # density_axis = [self.get_density(label) for label in label_axis]
        # color_axis = [ColorPalette[label] for label in label_axis]

        # ci = kwds.pop('ci', False)
        # yerr = [self.get_ci(label) for label in label_axis] if ci else None

        # ax.bar(x_axis, density_axis, width=0.7, color=color_axis, lw=2, edgecolor='k', yerr=yerr)
        
        # ax.set_xticks(x_axis)
        # ax.set_xticklabels(label_axis)
        # ax.set_ylabel('density')
        # ax.set_yticks([0, 0.5, 1])

        # # for x, y in zip(x_axis, density_axis):
        # #     ax.text(x, y + 0.01, f'{y:.2f}', color='k', ha='center', va='bottom', fontsize=16)

        ax = kwds.pop('ax', None)
        if ax is None:
            ax = prepare_canvas()
        
        ax.invert_yaxis()
        # ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.25, 0.6)

        label_axis = self.labels
        density_axis = np.array([self.get_density(label) for label in label_axis])
        data_cum = density_axis.cumsum(axis=0)
        color_axis = [ColorPalette[label] for label in label_axis]

        for i, color in enumerate(color_axis):
            widths = density_axis[i]
            starts = data_cum[i] - widths
            rects = ax.barh(0, widths, left=starts, height=0.4, align='center',
                            color=color, alpha=1, lw=2, edgecolor='k')

            # r, g, b, _ = color
            # text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
            ax.bar_label(rects, label_type='center', color='k', fmt='%.2f', size=14)

        tick_positions = data_cum - density_axis / 2
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(label_axis)

        # x_axis = np.arange(len(self.labels))

        # ci = kwds.pop('ci', False)
        # yerr = [self.get_ci(label) for label in label_axis] if ci else None

        # plot a horizontal bar chart for density_axis
        # ax.barh(x_axis, density_axis, height=0.7, color=color_axis, lw=2, edgecolor='k', xerr=yerr, alpha=0.5)
        # ax.bar(x_axis, density_axis, width=0.7, color=color_axis, lw=2, edgecolor='k', yerr=yerr, alpha=0.5)
        
        # ax.set_xticks(x_axis)
        # ax.set_xticklabels(label_axis)
        # ax.set_ylabel('density')
        # ax.set_xticks([0, 0.5, 1])
        # for Figma fig
        # hide y ticks
        # ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)


class MultinomialDUD(DiscreteUnivariateDistribution):
    """A class for parametric multinomial discrete univariate distributions.
    
    `MultinomialDUD` is a class to generate discrete data samples
    of a random variable from a collection of parametric
    sub-distributions with weights.
    
    Parameters
    ----------
    labels : list
        Labels of the multinomial distribution
    
    probs : list
        Non-negative probabilities for the labels

    Methods
    ----------
    generate_data

    Examples
    --------
    If you want to generate a multinomial distribution consisting of two
    classes and the weights for each class are 2:1, you can do the following:

    >>> mdud_rv = MultinomialDUD(labels=['neg', 'pos'], probs=[2, 1])

    You can sample 10,000 data points from the multinomial distribution,

    >>> mdud_rv.generate_data(n=10000)

    You can obtain the probability density of any class

    >>> mdud_rv.get_density('pos')
    """
    def __init__(self, labels, probs):
        self.labels = labels
        self.probs = np.array(probs) / np.sum(probs)  # normalize the probs
        self.label_prob_dict = self.get_label_prob_dict()

    def generate_data(self, n):
        """Generate a simulated dataset of size `n`.

        Parameters
        ----------
        n : int
            Number of samples to generate
        
        Returns
        -------
        rvs : ndarray
            Random variates of given `size`
        """
        return np.random.choice(self.labels, size=n, replace=True, p=self.probs)


class BinnedDUD(DiscreteUnivariateDistribution, EmpiricalData):
    """A class for binned discrete univariate distributions.
    
    `BinnedDUD` is a class to bin discrete data samples by their labels.
    
    Parameters
    ----------
    data : list
        Data samples to be binned

    Methods
    ----------
    sample
    get_ci

    Examples
    --------
    If you want to generate a binned distribution, you can do the following:

    >>> bdud_rv = BinnedDUD(data=np.random.choice(['neg', 'pos'], size=10000, p=[0.7, 0.3]))

    You can sample without replacement 1000 data points from the original data,

    >>> bdud_rv.sample(n=1000)

    You can also bootstrap (sample with replacement) 1000 data points from the original data,

    >>> bcud_rv.sample(n=1000, replace=True)

    You can obtain the 95 confidence intervals of any class

    >>> bdud_rv.get_ci('pos')
    """
    def __init__(self, data):
        self.data = np.array(data)
        self.labels = list(set(data))
        self.probs = np.array([np.mean(self.data == label) for label in self.labels])
        self.label_prob_dict = self.get_label_prob_dict()

    # def sample(self, n, replace=False):
    #     """Sample random variates of given size `n`.
        
    #     Parameters
    #     ----------
    #     n : int
    #         Number of samples to generate
    #     replace : bool, optional
    #         Whether to sample with replacement or not
            
    #     Returns
    #     -------
    #     rvs : ndarray
    #         Random variates of given `size`
    #     """
    #     return np.random.choice(self.data, size=n, replace=replace)
    
    def get_ci(self, label: str):
        """95 confidence interval of a given label.

        Parameters
        ----------
        label : str
            Label to be evaluated

        Returns
        -------
        ci : float
            Confidence interval of the given label
        """
        p_label = self.get_density(label)
        num_item = sum(self.data == label)
        return 1.96 * np.sqrt(p_label * (1 - p_label) / num_item)


class ContinuousUnivariateDistribution(UnivariateDistribution):
    """A base class for continuous univariate distributions.
    """

    def plot(self, **kwds):
        """Plot the probability density function of continuous univariate distribution class.
        """
        ax = kwds.pop('ax', None)
        if ax is None:
            ax = prepare_canvas()

        prev_ymax = ax.get_ylim()[1]

        weight = kwds.pop('weight', 1)
        bottom_axis = kwds.pop('bottom_axis', None)
        if bottom_axis is None:
            bottom_axis = np.zeros(len(self.x_axis))

        top_axis = bottom_axis + weight * self.y_axis

        one_gradient_plot(ax, self.x_axis, top_axis, bottom_axis, **kwds)

        ax.set_xlabel('score')
        ax.set_ylabel('density')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim(ymin=0)
        # ax.set_ylim([0, max((np.max(top_axis) * 1.1, prev_ymax))])

        return_bottom = kwds.pop('return_bottom', False)
        return_ax = kwds.pop('return_ax', False)
        if return_bottom and return_ax:
            return top_axis, ax
        if return_bottom:
            return top_axis
        if return_ax:
            return ax


class MixtureCUD(ContinuousUnivariateDistribution):
    """A class for parametric mixture continuous univariate distributions.

    `MixtureCUD` is a class to generate continuous data samples
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
    component are 2:7:1, you can do the following:

    >>> from scipy.stats import beta, uniform
    >>> mcud_rv = MixtureCUD(components=[beta(8, 2), beta(2, 5), uniform(0, 1)],
    ...                      weights=[2, 7, 1])

    You can sample 10,000 data points from the mixture distribution,

    >>> mcud_rv.generate_data(n=10000)

    You can obtain the probability density at any x

    >>> mcud_rv.get_density(0.7)

    """
    def __init__(self, components, weights):
        self.components = components
        self.num_component = len(components)
        weight_sum = sum(weights)
        self.weights = [weight / weight_sum for weight in weights]
        self.update_bin_axis(num_bin=1000)

    def update_bin_axis(self, num_bin):
        """Update the bin axis for plotting because we can generate infinite data from theoretical distributions.

        Parameters
        ----------
        num_bin : int
            Number of bins
        """
        bin_width = 1 / num_bin
        bin_margin = bin_width / 2
        # self.x_axis = np.arange(bin_margin, 1, bin_width)
        self.x_axis = np.linspace(0, 1, num_bin+1)
        self.y_axis = np.array([self.get_density(x) for x in self.x_axis])
        self.num_bin = len(self.x_axis)

    def get_density(self, score):
        """Probability density at `score`.

        Parameters
        ----------
        score : float
            Score at which the probability density is evaluated

        Returns
        -------
        pdf : ndarray
            Probability density evaluated at x
        """
        return sum([weight * component.pdf(score) if hasattr(component, 'pdf') and callable(component.pdf) else weight * component.get_density(score)
                    for weight, component in
                    zip(self.weights, self.components)])

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
        """Plot the probability density function of mixture continuous univariate distribution class.
        When plotting the theoretical distributions, we use a very large number of bins 
        because we can calculate the density at any score.
        
        Parameters
        ----------
        num_bin : int, optional
            Number of bins for plotting
        """
        num_bin = kwds.pop('num_bin', 1000)
        self.update_bin_axis(num_bin=num_bin)
        return super().plot(**kwds)


class BinnedCUD(ContinuousUnivariateDistribution, EmpiricalData):
    """A class for binned continuous univariate distributions.
    
    `BinnedCUD` is a class to bin continuous data samples by their values.
    
    Parameters
    ----------
    data : list
        Data samples to be binned
    
    num_bin : int
        Number of bins, this is required to model the empirical data distribution

    Methods
    ----------
    sample
    get_density

    Examples
    --------
    If you want to generate a binned distribution, you can do the following:

    >>> from scipy.stats import beta, uniform
    >>> bcud_rv = BinnedCUD(data=beta(8, 2).rvs(size=10000))

    You can sample without replacement 1000 data points from the original data,

    >>> bcud_rv.sample(n=1000)

    You can also bootstrap (sample with replacement) 1000 data points from the original data,

    >>> bcud_rv.sample(n=1000, replace=True)
    
    You can obtain the probability density at any x

    >>> bcud_rv.get_density(0.7)
    """
    def __init__(self, data=None, num_bin=10, x_axis=None, y_axis=None):
        if data is not None:
            self.data = np.array(data)
            self.num_bin = num_bin
            bin_width = 1 / num_bin
            bin_margin = bin_width / 2
            self.x_axis = np.arange(bin_margin, 1, bin_width)
            # self.x_axis = np.linspace(0, 1, num_bin+1)
            bin_edges = np.linspace(0, 1, self.num_bin + 1)
            self.y_axis = np.histogram(self.data, bins=bin_edges, density=True)[0]
        else:
            self.x_axis = x_axis
            self.y_axis = y_axis
            self.num_bin = len(self.x_axis)

    def get_density(self, score):
        """Probability density at `score`.

        Parameters
        ----------
        score : float
            Score at which the probability density is evaluated

        Returns
        -------
        pdf : ndarray
            Probability density evaluated at `score`
        """
        # find the index of element in x_axis that is closest to score
        # return self.y_axis[np.searchsorted(self.x_axis, score)]
        return self.y_axis[np.argmin(np.abs(self.x_axis - score))]


class JointDistribution:
    """A base class for joint distributions.
    
    `JointDistribution` is a class to model joint distribtions.
    
    Parameters
    ----------
    labels : list
        Labels of the joint distribution

    Methods
    ----------
    sample
    get_density
    plot_five_distributions
    """
    def __init__(self, labels):
        self.labels = labels
        self.label_distribution = None
        self.class_conditional_densities = None
        self.classifier_score_distribution = None
        self.calibration_curve = None

    def sample(self):
        pass

    def get_density(self):
        pass

    def plot_five_distributions(self, num_bin=1000):
        """Plot the five distributions of a joint distribution.

        Parameters
        ----------
        num_bin : int
            Number of bins to use
        """
        fig, axes = plt.subplots(1, 5, figsize=(16, 3))
        axes = axes.ravel()

        bottom_line = None
        top_y_axis = 0
        for label in self.labels:
            bottom_line = self.class_conditional_densities[label].plot(ax=axes[0], 
                                                                       num_bin=num_bin,
                                                                       color=ColorPalette[label],
                                                                       return_bottom=True)
            top_y_axis = max(top_y_axis, np.max(bottom_line))
        axes[0].set_ylim(top=top_y_axis)
        axes[0].set_title('Class Conditional Densities')

        self.label_distribution.plot(ax=axes[1])
        axes[1].spines['left'].set_visible(False)
        axes[1].set_title('Label Density')

        cum_bottom_line = None
        for label in self.labels:
            weight = self.label_distribution.get_density(label)
            cum_bottom_line = self.class_conditional_densities[label].plot(ax=axes[2], 
                                                                           num_bin=num_bin, 
                                                                           color=ColorPalette[label], 
                                                                           return_bottom=True, 
                                                                           bottom_axis=cum_bottom_line,
                                                                           weight=weight)
        axes[2].set_ylim(top=np.max(cum_bottom_line))
        axes[2].set_title('Joint Density')

        self.classifier_score_distribution.plot(ax=axes[3], num_bin=num_bin)
        axes[3].set_title('Classifier Score Density')

        if hasattr(self, 'update_calibration_curve') and callable(self.update_calibration_curve):
            self.update_calibration_curve(num_bin=num_bin)
        self.calibration_curve.plot(ax=axes[4])
        axes[4].set_title('Calibration Curve')

        for ax in axes:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='both', which='major')
        
        plt.tight_layout()
        # #save plt as pdf
        # plt.savefig('five_distributions.pdf')
        return axes


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
        # x_axis = np.arange(0.5/num_bin, 1, 1/num_bin)
        x_axis = np.linspace(0, 1, num_bin+1)
        pos_weight = self.label_distribution.get_density('pos')
        y_axis = [pos_weight * self.class_conditional_densities['pos'].get_density(x) /
                  self.classifier_score_distribution.get_density(x)
                  for x in x_axis]
        return BinnedCalibrationCurve(x_axis, y_axis)
    
    def update_calibration_curve(self, num_bin):
        self.calibration_curve = self.calculate_calibration_curve(num_bin=num_bin)


class ExtrinsicJointDistribution(JointDistribution):
    def __init__(self, labels: list,
                 classifier_score_distribution: ContinuousUnivariateDistribution,
                 calibration_curve: CalibrationCurve):
        super().__init__(labels)
        self.classifier_score_distribution = classifier_score_distribution
        self.calibration_curve = calibration_curve

        num_bin = classifier_score_distribution.num_bin
        self.label_distribution = self.calculate_label_distribution(num_bin)
        self.class_conditional_densities = self.calculate_class_conditional_densities(num_bin)

    def calculate_label_distribution(self, num_bin):
        x_axis = np.arange(0.5/num_bin, 1, 1/num_bin)
        # x_axis = np.linspace(0, 1, num_bin+1)
        area_pos = np.nansum(self.calibration_curve.get_calibrated_prob(x_axis) * \
                    np.array([self.classifier_score_distribution.get_density(x)
                            for x in x_axis]))
        area_neg = np.nansum((1 - self.calibration_curve.get_calibrated_prob(x_axis)) * \
                    np.array([self.classifier_score_distribution.get_density(x)
                            for x in x_axis]))
        total_area = area_pos + area_neg
        area_pos /= total_area
        area_neg /= total_area
        return MultinomialDUD(['neg', 'pos'], np.array([area_neg, area_pos]))

    def calculate_class_conditional_densities(self, num_bin):
        # x_axis = np.arange(0.5/num_bin, 1, 1/num_bin)
        x_axis = np.linspace(0, 1, num_bin+1)
        curve_pos = self.calibration_curve.get_calibrated_prob(x_axis) * \
                    np.array([self.classifier_score_distribution.get_density(x)
                            for x in x_axis])
        curve_neg = (1 - self.calibration_curve.get_calibrated_prob(x_axis)) * \
                    np.array([self.classifier_score_distribution.get_density(x)
                            for x in x_axis])
        curve_pos = np.array(curve_pos) / sum(curve_pos) * num_bin
        curve_neg = np.array(curve_neg) / sum(curve_neg) * num_bin
        return {'pos': BinnedCUD(x_axis=x_axis, y_axis=curve_pos),
                'neg': BinnedCUD(x_axis=x_axis, y_axis=curve_neg)}
