import numpy as np
import pandas as pd
from pyquantifier.distributions import BinnedDUD, BinnedCUD
from sklearn.linear_model import LogisticRegression
from pyquantifier.calibration_curve import PlattScaling


# create an Item class
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


# create a Dataset class
class Dataset:
    def __init__(self, labels, df=None, items=None):
        self.labels = labels
        if df is not None:
            self.df = df
        elif items is not None:
            self.df = self.to_dataframe(items)
        else:
            raise ValueError('either df or items must be provided')

    @staticmethod
    def to_dataframe(items):
        return pd.concat([item.to_row() for item in items], axis=1).T

    def to_csv(self, path):
        self.df.to_csv(path, index=False)

    def sample(self, n):
        # sample without replacement n items from self.df
        return Dataset(df=self.df.sample(n, replace=False), labels=self.labels)

    def get_sample_for_annotation(self, n, strategy='random'):
        # only return the row indices that are needed for annotation
        if strategy == 'random':
            return Dataset(df=self.df.sample(n, replace=False),
                           labels=self.labels)
        elif strategy == 'uniform':
            pass  # TODO
        elif strategy == 'neyman':
            pass  # TODO
        else:
            raise ValueError('unsupported sampling strategy, '
                             'options are random, uniform, or neyman.')

    def annotate_sample(self, gt_labels):
        # annotate the sampled items with gt_labels
        self.df['gt_label'] = gt_labels

    def plot(self):
        pass

    def get_label_distribution(self):
        return BinnedDUD(self.df['gt_label'].tolist())

    def get_classifier_score_distribution(self):
        return BinnedCUD(self.df['pos'].tolist())

    def get_class_conditional_density(self, label):
        return BinnedCUD(self.df[self.df['gt_label'] == label]['pos'].tolist())

    def get_class_conditional_densities(self):
        return {label: self.get_class_conditional_density(label)
                for label in self.labels}

    def generate_calibration_curve(self, method='platt_scaling'):
        if method == 'platt_scaling':
            train_CX = self.df['pos'].values.reshape(-1, 1)
            train_GT = self.df['gt_label'].map({'neg': 0, 'pos': 1}).values
            prob_cali_func = LogisticRegression(solver='lbfgs', fit_intercept=True).fit(train_CX, train_GT)
            prob_cali_obj = PlattScaling()
            prob_cali_obj.set_params(prob_cali_func.coef_,
                                     prob_cali_func.intercept_)
            return prob_cali_obj


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
test_dataset()
