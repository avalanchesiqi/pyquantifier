from collections import Counter

import numpy as np

from pyquantifier.plot import plot_empirical_bar


class ParametricLabelDensity:
    """Parametric Label density
    """
    def __init__(self, class_prevalences, class_names):
        self.num_class = len(class_names)
        class_prevalences = np.array(class_prevalences) / sum(class_prevalences)
        self.class_prevalence_dict = {class_name: class_prevalence
                                      for class_name, class_prevalence in
                                      zip(class_names, class_prevalences)}

    def get_class_prevalence(self, class_name):
        """Obtain the prevalence for a given `label`
        """
        return self.class_prevalence_dict[class_name]

    def plot_bar(self, **kwds):
        plot_empirical_bar(self.class_prevalence_dict, **kwds)


# class LabelDensityLookupTable(LabelDensity):
#     def __int__(self):
#         super().__int__()

class InferredLabelDensity:
    """Inferred Label density

    `LabelDensity` is a class to store prevalence of GT labels.

    Parameters
    ----------
    labels : list
        List of data point labels.

    Attributes
    ----------
    label_counter : dict
        Counter of each label.

    size : int
        Size of the data labels.

    norm_label_counter : dict
        Normalized counter of each label.

    Methods
    ----------
    get_prevalence
    plot_bar

    Examples
    --------
    """
    def __init__(self, labels):
        self.label_counter = Counter(labels)
        self.size = len(labels)
        self.norm_label_counter = {k: v/self.size
                                   for k, v in self.label_counter.items()}

    def get_prevalence(self, label):
        """Obtain the prevalence for a given `label`
        """
        return self.norm_label_counter[label]

    def plot_bar(self, **kwds):
        plot_empirical_bar(self.norm_label_counter, **kwds)