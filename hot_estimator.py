import os, sys, json, pickle
import numpy as np
import pandas as pd
from collections import Counter
from pyquantifier.data import Dataset


# generate calibration curve
def generate_calibration_curve():
    uid_list = []
    pos_list = []
    neg_list = []
    gt_label_list = []
    uid = 1

    all_labels = ['pos', 'neg']
    label_map = {True: 'pos', False: 'neg'}
    with open('labeled_hot_data.json', 'r') as fin:
        for line in fin:
            comment_json = json.loads(line.rstrip())
            toxicity = comment_json['new_toxicity']
            toxic_label = label_map[Counter([x for x, y in comment_json['composite_toxic']]).most_common(1)[0][0]]
            uid_list.append(uid)
            pos_list.append(toxicity)
            neg_list.append(1-toxicity)
            gt_label_list.append(toxic_label)
            uid += 1

    hot_df = pd.DataFrame.from_dict({'uid': uid_list, 'pos': pos_list, 'neg': neg_list, 'gt_label': gt_label_list})
    hot_dataset = Dataset(df=hot_df, labels=all_labels)

    calibration_curve = hot_dataset.generate_calibration_curve(method='platt scaling')
    pickle.dump(calibration_curve, open('calibration_curve.pkl', 'wb'))


def main():
    if not os.path.exists('calibration_curve.pkl'):
        generate_calibration_curve()
    calibration_curve = pickle.load(open('calibration_curve.pkl', 'rb'))

    # TODO: load the toxicity scores of a day, as a list
    cx_list = np.random.rand(10000).tolist()
    num_items = len(cx_list)

    # build a dataset object from the list
    df = pd.DataFrame.from_dict({'uid': list(range(num_items)), 'pos': cx_list, 'neg': 1-np.array(cx_list)})
    dataset = Dataset(df=df, labels=['pos', 'neg'])

    prevalence_est = dataset.extrinsic_estimate(calibration_curve=calibration_curve)
    print(f'use platt scaling: {prevalence_est:.4f} on the a simulated data')


if __name__ == '__main__':
    main()