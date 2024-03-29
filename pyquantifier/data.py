import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyquantifier.plot import *

from pyquantifier.distributions import BinnedDUD, BinnedCUD, ExtrinsicJointDistribution, IntrinsicJointDistribution
from sklearn.linear_model import LogisticRegression
from pyquantifier.calibration_curve import PlattScaling, BinnedCalibrationCurve, CalibrationCurve
from pyquantifier.quantifier.intrinsic_estimator import MixtureModelEstimator
from pyquantifier.util import get_bin_idx


# Item class
class Item:
    def __init__(self, uid, labels, probs, gt_label=None, **feature_kwargs):
        self.uid = uid
        self.labels = labels
        self.probs = np.array(probs) / np.sum(probs)  # normalize the probs
        self.gt_label = gt_label
        self.__dict__.update(feature_kwargs)

    def set_gt_label(self, gt_label):
        self.gt_label = gt_label

    def prob_correct(self):
        # the probability of correct is the probability of gt_label
        if self.gt_label:
            return self.probs[self.labels.index(self.gt_label)]
        else:
            raise ValueError('gt_label is not set')

    def prob_incorrect(self):
        # the probability of incorrect is the probability of not gt_label
        return 1 - self.prob_correct()

    def to_row(self):
        # convert the item to a pandas Series
        item_dict = {'uid': self.uid}
        for label, prob in zip(self.labels, self.probs):
            item_dict[label] = prob
        for feature, feature_val in self.__dict__.items():
            if feature not in ['uid', 'labels', 'probs', 'gt_label']:
                item_dict[feature] = feature_val
        return pd.Series(item_dict)


# Dataset class
class Dataset:
    def __init__(self, labels, df=None, items=None):
        """Initialize a Dataset object.

        Parameters
        ----------
        labels : list
            A list of possible labels
        df : DataFrame
            A pandas DataFrame object
        items : list
            A list of Item objects
        """
        self.labels = labels
        if df is not None:
            self.df = df
        elif items is not None:
            self.df = self.to_dataframe(items)
        else:
            raise ValueError('either df or items must be provided')
        self.label_distribution = None
        self.class_conditional_densities = None
        self.classifier_score_distribution = None
        self.calibration_curve = None
    
    def update_dataset_model(self, num_bin, selection_weights=None):
        self.class_conditional_densities = self.infer_class_conditional_densities(num_bin, selection_weights)
        self.label_distribution = self.infer_label_distribution()
        self.classifier_score_distribution = self.infer_classifier_score_distribution(num_bin)
        # if isinstance(self.calibration_curve, BinnedCalibrationCurve):
        # self.calibration_curve = self.generate_calibration_curve('nonparametric binning', num_bin=num_bin)
        self.calibration_curve = self.generate_calibration_curve('platt scaling', num_bin=num_bin)

    def update_calibration_curve(self, **kwds):
        self.calibration_curve = self.generate_calibration_curve(**kwds)

    @staticmethod
    def to_dataframe(items):
        """Convert a list of items to a pandas DataFrame.

        Parameters
        ----------
        items : list
            A list of Item objects
        
        Returns
        -------
        DataFrame
            A pandas DataFrame object
        """
        return pd.concat([item.to_row() for item in items], axis=1).T

    def to_csv(self, path):
        """Save the dataset to a csv file.
        
        Parameters
        ----------
        path : str
            Path to save the dataset
        """
        self.df.to_csv(path, index=False)

    def sample(self, n, replace=False):
        """Sample n items from the dataset without replacement 
        or bootstrap n items from the dataset with replacement

        Parameters
        ----------
        n : int
            Number of items to sample
        replace : bool
            Whether to sample with replacement
        
        Returns
        -------
        Dataset
            A new dataset object with n items
        """
        return Dataset(df=self.df.sample(n, replace=replace), labels=self.labels)
    
    @staticmethod
    def _get_neyman_allocation(bin_dict, total_n):
        """Get the neyman allocation for each bin.

        Parameters
        ----------
        bin_dict : dict
            A dict of number of items for each bin
        total_n : int
            Total number of items to sample
        
        Returns
        -------
        dict
            A dictionary of bin index and number of items to sample from each bin
        """
        n = len(bin_dict)
        N = np.array([bin_dict[i] for i in range(n)])
        step = 1 / n
        K = np.arange(step/2, 1, step)
        S = np.sqrt(K * (1 - K))
        return {i: round(n_in_bin) for i, n_in_bin in enumerate(total_n * (N * S) / sum(N * S))}

    @staticmethod    
    def _sample_items_from_bin(df, bin_dict):
        """Sample varying numbers of items based on bin_dict.

        Parameters
        ----------
        df : DataFrame
            A pandas DataFrame object
        bin_dict : dict
            A dictionary of bin index and number of items to sample from each bin
        
        Returns
        -------
        DataFrame
            A pandas DataFrame object
        """
        return df.sample(n=int(bin_dict[df['bin'].iloc[0]]), replace=True)

    def select_sample_for_annotation(self, n, strategy='random', bins=10):
        """Select n items from the dataset for annotation.

        Parameters
        ----------
        n : int
            Number of items to select
        strategy : str
            Sampling strategy, options are random, uniform, or neyman.
        bins : int
            Number of bins to use for uniform or neyman sampling
        
        Returns
        -------
        Dataset
            A new dataset object with n items
        """

        if strategy == 'random':
            return self.sample(n), None
        elif strategy in ['uniform', 'neyman']:
            df = self.df.copy()
            # create a new column based on the pos column
            df['bin'] = df.apply(lambda row: get_bin_idx(row['pos'], size=bins), axis=1)
            original_bin_dict = df['bin'].value_counts().to_dict()
            for i in range(bins):
                if i not in original_bin_dict:
                    original_bin_dict[i] = 0

            if strategy == 'uniform':
                n_per_bin = n // bins
                to_sample_bin_dict = {i: n_per_bin for i in range(bins)}
            else:  # neyman
                to_sample_bin_dict = self._get_neyman_allocation(original_bin_dict, n)

            # sample items from each bin
            df = df.groupby(by='bin', sort=True, group_keys=True)\
                .apply(self._sample_items_from_bin, bin_dict=to_sample_bin_dict)
            # drop the bin column
            df = df.drop(columns=['bin'])

            print('original_bin_dict', original_bin_dict)
            print('to_sample_bin_dict', to_sample_bin_dict)
            selection_weights = [to_sample_bin_dict[i] / original_bin_dict[i] for i in range(bins)]
            return Dataset(df=df, labels=self.labels), selection_weights
        else:
            raise ValueError('unsupported sampling strategy, '
                            'options are random, uniform, or neyman.')

    def annotate_sample(self, gt_labels):
        """Annotate the sampled items with gt_labels.

        Parameters
        ----------
        gt_labels : list
            A list of ground truth labels
        """
        # annotate the sampled items with gt_labels
        self.df['gt_label'] = gt_labels
    
    def infer_label_distribution(self):
        """Infer the label distribution from the dataset.

        Returns
        -------
        BinnedDUD
            A BinnedDUD object
        """
        return BinnedDUD(self.df['gt_label'].tolist())

    def infer_classifier_score_distribution(self, num_bin):
        """Infer the classifier score distribution from the dataset.

        Parameters
        ----------
        num_bin : int
            Number of bins to use

        Returns
        -------
        BinnedDUD
            A BinnedDUD object
        """
        x_axis = np.arange(0.5 / num_bin, 1, 1 / num_bin)
        y_axis, dummy = np.histogram(self.df['pos'].tolist(), bins=np.linspace(0, 1, num_bin+1), density=True)
        return BinnedCUD(x_axis=x_axis, y_axis=y_axis)

    def infer_class_conditional_density(self, label, num_bin, selection_weights=None):
        """Infer the class conditional density for a given label class.

        Parameters
        ----------
        label : str
            A label class
        num_bin : int
            Number of bins to use

        Returns
        -------
        BinnedDUD
            A BinnedDUD object
        """
        x_axis = np.arange(0.5 / num_bin, 1, 1 / num_bin)
        y_axis, dummy = np.histogram(self.df[self.df['gt_label'] == label]['pos'].tolist(), bins=np.linspace(0, 1, num_bin+1), density=True)
        if selection_weights is not None:
            assert num_bin == len(selection_weights)
            selection_weights = np.array(selection_weights)
            y_axis /= selection_weights
            y_axis *= num_bin / y_axis.sum()
        return BinnedCUD(x_axis=x_axis, y_axis=y_axis)

    def infer_class_conditional_densities(self, num_bin, selection_weights):
        """Infer the class conditional densities for all label classes.

        Returns
        -------
        dict
            A dictionary of BinnedDUD objects
        """
        return {label: self.infer_class_conditional_density(label, num_bin, selection_weights)
                for label in self.labels}

    def generate_calibration_curve(self, method='platt scaling', num_bin=10):
        """Generate a calibration curve for the dataset.

        Parameters
        ----------
        method : str
            Calibration method, options are platt scaling, temperature scaling, or nonparametric binning.
        num_bin : int
            Number of bins to use for nonparametric binning

        Returns
        -------
        CalibrationCurve
            A CalibrationCurve object
        """
        if method == 'platt scaling':
            train_CX = self.df['pos'].values.reshape(-1, 1)
            train_GT = self.df['gt_label'].map({'neg': 0, 'pos': 1}).values
            prob_cali_func = LogisticRegression(solver='lbfgs', fit_intercept=True).fit(train_CX, train_GT)

            prob_cali_obj = PlattScaling()
            prob_cali_obj.set_params(prob_cali_func.coef_, prob_cali_func.intercept_)
            return prob_cali_obj
        elif method == 'temperature scaling':
            pass
        elif method == 'nonparametric binning':
            x_axis = np.arange(0.5/num_bin, 1, 1/num_bin)
            df = self.df.copy()
            # create a new column based on the pos column
            df['bin'] = df.apply(lambda row: get_bin_idx(row['pos'], size=num_bin), axis=1)
            y_axis = [len(df[(df['bin']==bin_idx) & (df['gt_label']=='pos')]) / len(df[df['bin']==bin_idx]) for bin_idx, _ in enumerate(x_axis)]
            prob_cali_obj = BinnedCalibrationCurve(x_axis=x_axis, y_axis=y_axis)
            return prob_cali_obj
        else:
            raise ValueError('unsupported calibration method, '
                            'options are platt scaling, temperature scaling, or nonparametric binning.')
    
    # def infer_joint_density(self):
    #     """Infer the joint density of the dataset.

    #     Returns
    #     -------
    #     ExtrinsicJointDistribution
    #         An ExtrinsicJointDistribution object
    #     """
    #     return ExtrinsicJointDistribution(self.labels,
    #                                       self.infer_classifier_score_distribution(num_bin=10),
    #                                       self.generate_calibration_curve('platt scaling'))

    def profile_dataset(self, num_bin=10, selection_weights=None):
        """Plot the five distributions of the dataset.

        Parameters
        ----------
        num_bin : int
            Number of bins to use
        """
        fig, axes = plt.subplots(1, 5, figsize=(16, 3))
        axes = axes.ravel()

        self.update_dataset_model(num_bin=num_bin, selection_weights=selection_weights)

        for label in self.labels:
            self.class_conditional_densities[label].plot(ax=axes[0], color=ColorPalette[label])
        # axes[0].set_title('Class Conditional Densities')

        self.label_distribution.plot(ax=axes[1], ci=False)
        axes[1].spines['left'].set_visible(False)
        # axes[1].set_title('Label Density')

        prev_bottom = None
        for label in self.labels:
            weight = self.label_distribution.get_density(label)
            prev_bottom = self.class_conditional_densities[label].plot(
                ax=axes[2], bottom_axis=prev_bottom, color=ColorPalette[label], 
                return_bottom=True, weight=weight)
        # axes[2].set_title('Joint Density')

        self.classifier_score_distribution.plot(ax=axes[3], density=True)
        # axes[3].set_title('Classifier Score Density')

        self.calibration_curve.plot(ax=axes[4], show_diagonal=False)
        # axes[4].set_title('Calibration Curve')

        for ax in axes:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='both', which='major')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
        
        plt.tight_layout()
        return axes
    
    def profile_dataset_rev(self, num_bin=10, selection_weights=None):
        """Plot the five distributions of the dataset.

        Parameters
        ----------
        num_bin : int
            Number of bins to use
        """
        fig, axes = plt.subplots(1, 5, figsize=(16, 3))
        axes = axes.ravel()

        self.update_dataset_model(num_bin=num_bin, selection_weights=selection_weights)

        for label in self.labels:
            self.class_conditional_densities[label].plot(ax=axes[3], color=ColorPalette[label])
        # axes[0].set_title('Class Conditional Densities')

        self.label_distribution.plot(ax=axes[4], ci=False)
        axes[4].spines['left'].set_visible(False)
        # axes[1].set_title('Label Density')

        prev_bottom = None
        for label in self.labels:
            weight = self.label_distribution.get_density(label)
            prev_bottom = self.class_conditional_densities[label].plot(
                ax=axes[2], bottom_axis=prev_bottom, color=ColorPalette[label], 
                return_bottom=True, weight=weight)
        # axes[2].set_title('Joint Density')

        self.classifier_score_distribution.plot(ax=axes[1], density=True)
        # axes[3].set_title('Classifier Score Density')

        self.calibration_curve.plot(ax=axes[0], show_diagonal=False)
        # axes[4].set_title('Calibration Curve')

        for ax in axes:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='both', which='major')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')

        plt.tight_layout()
        return axes


    def intrinsic_estimate(self, class_conditional_densities: dict, method='mixture model'):
        if method == 'mixture model':
            prevalence_estimator = MixtureModelEstimator()
            prevalence_estimator.set_positive_density(class_conditional_densities['pos'])
            prevalence_estimator.set_negativity_density(class_conditional_densities['neg'])

            est_prev = prevalence_estimator.estimate(self.df['pos'].values)
            return est_prev

    def extrinsic_estimate(self, calibration_curve: CalibrationCurve):
        self.df['cali_pos'] = calibration_curve.get_calibrated_prob(self.df['pos'].values)
        return self.df['cali_pos'].sum() / len(self.df)


# generate a unit test for the Dataset class
def test_dataset():
    # create an item
    item1 = Item(uid='p1',
                 labels=['pos', 'neg'],
                 probs=[0.2, 0.8],
                 gender='female',
                 age='young')

    # create more items similar to item1
    item2 = Item(uid='p2',
                 labels=['pos', 'neg'],
                 probs=[0.3, 0.7],
                 gender='female',
                 age='old')

    item3 = Item(uid='p3',
                 labels=['pos', 'neg'],
                 probs=[0.9, 0.1],
                 gender='male',
                 age='old')

    # create a dataset
    dataset = Dataset(labels=['pos', 'neg'], items=[item1, item2, item3])
    print(dataset.df)

    # test the dataset
    assert dataset.df.shape == (3, 5)


# run the unit test for Dataset class
# test_dataset()
