# -*- coding: utf-8 -*-

import numpy as np
from pyquantifier.conf import *
from pyquantifier.plot import prepare_canvas, one_gradient_plot, \
    save_to_img, plot_empirical_hist, fine_axis, ColorPalette


class MixtureDistribution:
    """A theoretical, parametric mixture distribution class.

    `MixtureDistribution` is a base class to generate continuous data samples
    of a random variable from a collection of theoretical parametric
    sub-distributions with weights.

    Parameters
    ----------
    components : list
        Theoretical parametric sub-distributions (mixture components) to
        constitute the mixture distribution.

    weights : list
        Non-negative weights for the mixture components.

    Attributes
    ----------
    components : list
        Theoretical parametric sub-distributions.

    num_component : int
        The number of the mixture components.

    weights : list
        Non-negative weights for the mixture components that sum to 1.

    Methods
    ----------
    pdf
    rvs
    plot_pdf
    plot_sample_hist

    Examples
    --------
    If you want to generate a mixture distribution consisting of two Beta
    distributions and one uniform distribution, and the weights for each
    compoment are 2:7:1, you can do the following:

    >>> from scipy.stats import beta, uniform
    >>> md_rv = MixtureDistribution([beta(8, 2), beta(2, 5), uniform(0, 1)],
    ...                              weights=[2, 7, 1])

    You can sample 10,000 data points from the mixture distribution,

    >>> md_rv.rvs(size=10000)

    You can obtain the probability density function at any x

    >>> import numpy as np
    >>> md_rv.pdf(np.linspace(0, 1, 101))

    """

    def __init__(self, components, weights):
        self.components = components
        self.num_component = len(components)
        weight_sum = sum(weights)
        self.weights = [weight / weight_sum for weight in weights]

    def pdf(self, x):
        """Probability density function at `x` of the given RV.

        Parameters
        ----------
        x : array_like
            quantiles

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at x

        """
        return sum([weight * component.pdf(x)
                    for weight, component in
                    zip(self.weights, self.components)])

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
        component_choices = np.random.choice(range(self.num_component),
                                             size=size,
                                             p=self.weights)
        component_samples = [component.rvs(size=size)
                             for component in self.components]
        data_sample = np.choose(component_choices, component_samples)
        return data_sample

    def plot_pdf(self, **kwds):
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

        rv_pdf = self.pdf(fine_axis)
        one_gradient_plot(ax, fine_axis, rv_pdf, **kwds)

        ax.set_xlabel('$C(X)$')
        ax.set_ylabel('$P(C(X))$')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim(ymin=0)
        save_to_img(img_name)

    def plot_hist(self, size, **kwds):
        """Plot the histogram for `size` data sample from the mixture
        distribution.

        Parameters
        ----------
        size : int
            Number of random variates
        """
        rv_scores = self.rvs(size)
        plot_empirical_hist(rv_scores, **kwds)
