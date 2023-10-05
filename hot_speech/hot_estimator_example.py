import os, sys, json, pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from pyquantifier.data import Dataset


def main():
    # ------- Start TODO ---------- #
    # TODO: load the toxicity scores of a day, as a list
    cx_list = np.random.rand(10000).tolist()
    # ------- END TODO ---------- #

    # build a dataset object from the list
    num_items = len(cx_list)
    df = pd.DataFrame.from_dict({'uid': list(range(num_items)), 'pos': cx_list, 'neg': 1-np.array(cx_list)})
    dataset = Dataset(df=df, labels=['pos', 'neg'])

    # load the calibration curve of the hot speech dataset
    cached_dists_filepath = 'hot_speech/hot_cached_dists.pkl'
    cached_dists = pickle.load(open(cached_dists_filepath, 'rb'))

    platform = 'reddit'  # platform can be 'reddit' or 'twitter' or 'youtube'
    metric = 'hot'  # metric can be 'hate', 'offensive', 'toxic', 'hot'

    calibration_curve = cached_dists[f'{platform}_{metric}_calibration_curve']
    ex_prevalence_est = dataset.extrinsic_estimate(calibration_curve=calibration_curve)
    print(f'extrinsic estimate: {ex_prevalence_est:.4f} on the a simulated data')

    # class_conditional_densities = cached_dists[f'{metric}_class_conditional_densities']
    # in_prevalence_est = dataset.intrinsic_estimate(class_conditional_densities=class_conditional_densities)
    # print(f'intrinsic estimate: {in_prevalence_est:.4f} on the a simulated data')

if __name__ == '__main__':
    main()
