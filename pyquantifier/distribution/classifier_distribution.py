from pyquantifier.plot import *


class ParametricClassifierDensity:
    """Parametric class conditional densities
    """
    def __init__(self, class_density):
        self.class_density = class_density

    def get_class_density_at_cx(self, cx):
        return self.class_density.pdf(cx)

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
        return self.class_density.rvs(size)

    def plot_pdf(self, **kwds):
        ax = kwds.pop('ax', None)
        if ax is None:
            ax = prepare_canvas()
        img_name = kwds.pop('img_name', None)

        rv_pdf = self.class_density.pdf(fine_axis)
        one_gradient_plot(ax, fine_axis, rv_pdf,
                          color=ColorPalette.unknown_color,
                          **kwds)

        ax.set_xlabel('$C(X)$')
        ax.set_ylabel('$P(C(X))$')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim(ymin=0)
        save_to_img(img_name)


class InferredClassifierDensity:
    """A non-parametric classifier score density.

    `ClassifierDensity` is a class to store continuous data samples
    of a random variable.

    Parameters
    ----------
    cxs : ndarray
        Numpy array of classifier scores.

    Attributes
    ----------
    cxs : ndarray
        Numpy array of classifier scores.

    size : int
        Size of the classifier scores.

    Methods
    ----------
    get_pdf
    plot_hist

    Examples
    --------
    """
    def __init__(self, cxs):
        self.cxs = np.array(cxs)
        self.size = len(self.cxs)

    def get_pdf(self, cx):
        """Obtain the pdf score for a given `cx`
        """
        return ((cx < self.cxs).sum() + (cx > self.cxs).sum()) / self.size / 2

    def plot_hist(self, **kwds):
        """Plot the empirical histogram for `cxs`
        """
        plot_empirical_hist(self.cxs, **kwds)


# class BetaDensity(ClassifierDensity):
#     def __int__(self):
#         super().__int__()
#
#
# class NonParametricDensity(ClassifierDensity):
#     def __int__(self):
#         super().__int__()
#
#
# class InferredClassifierDensity(ClassifierDensity):
#     def __int__(self):
#         super().__int__()
