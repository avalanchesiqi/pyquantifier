# -*- coding: utf-8 -*-

from collections import defaultdict
from pyquantifier.plot import *


class ParametricConditionalDensities:
    """Parametric class conditional densities
    """
    def __init__(self, class_densities, class_names):
        self.num_class = len(class_names)
        self.class_density_dict = {class_name: class_density
                                   for class_name, class_density in
                                   zip(class_names, class_densities)}

    def get_class_density_function(self, class_name):
        return self.class_density_dict[class_name]

    def get_class_density_at_cx(self, cx, class_name):
        return self.class_density_dict[class_name].pdf(cx)

    def plot_pdf(self, **kwds):
        cp = eval(f'ColorPalette.CC{self.num_class}')
        ax = kwds.pop('ax', None)
        if ax is None:
            ax = prepare_canvas()
        img_name = kwds.pop('img_name', None)

        for color_idx, (class_name, class_density) in enumerate(
                self.class_density_dict.items()):
            rv_pdf = class_density.pdf(fine_axis)
            one_gradient_plot(ax, fine_axis, rv_pdf,
                              color=cp[color_idx],
                              label=class_name, **kwds)

        ax.set_xlabel('$C(X)$')
        ax.set_ylabel('$P(C(X))$')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim(ymin=0)
        ax.legend(frameon=False)
        save_to_img(img_name)


class InferredConditionalDensities:
    """Inferred class conditional densities
    """
    def __init__(self, labeled_data):
        self.labeled_data = labeled_data

        self.class_densities = defaultdict(list)
        for cx, gt in labeled_data:
            self.class_densities[gt].append(cx)
        self.num_class = len(self.class_densities)

    def get_density(self, cx, label):
        pass

    def get_label_density_function(self, label):
        pass

    def plot_hist(self, **kwds):
        cp = eval(f'ColorPalette.CC{self.num_class}')
        for color_idx, (class_name, cxs) in enumerate(
                self.class_densities.items()):
            plot_empirical_hist(cxs, color=cp[color_idx],
                                density=True, ylabel='density', **kwds)


# class DictionaryOfConditionalDensities(ClassConditionalDensities):
#     def __int__(self):
#         super().__int__()
