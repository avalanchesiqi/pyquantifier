import os, sys, json, pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from pyquantifier.data import Dataset


def get_hot_values(toxicity_values, platform, metric):
    """
    `platform` can be 'reddit' or 'twitter' or 'youtube'
    `metric` can be 'hate', 'offensive', 'toxic', 'hot'
    """
    # build a dataset object from the list
    num_items = len(toxicity_values)
    df = pd.DataFrame.from_dict({
        'uid': list(range(num_items)),
        'pos': toxicity_values,
        'neg': 1-np.array(toxicity_values)
    })
    dataset = Dataset(df=df, labels=['pos', 'neg'])

    # load the calibration curve of the hot speech dataset
    cached_dists_filepath = 'hot_speech/hot_cached_dists.pkl'
    cached_dists = pickle.load(open(cached_dists_filepath, 'rb'))

    calibration_curve = cached_dists[f'{platform}_{metric}_calibration_curve']
    ex_prevalence_est = dataset.extrinsic_estimate(calibration_curve=calibration_curve)
    print(f'extrinsic estimate: {ex_prevalence_est:.4f} on the a simulated data')

    # class_conditional_densities = cached_dists[f'{metric}_class_conditional_densities']
    # in_prevalence_est = dataset.intrinsic_estimate(class_conditional_densities=class_conditional_densities)
    # print(f'intrinsic estimate: {in_prevalence_est:.4f} on the a simulated data')

    return ex_prevalence_est


if __name__ == '__main__':
    cx_list = np.random.rand(10000).tolist()
    print(get_hot_values(cx_list, 'reddit', 'hot'))
