import os, sys, json, pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter
import numpy as np
import pandas as pd

from pyquantifier.data import Dataset


def get_majority_vote(lst):
    return Counter(lst).most_common(1)[0][0]


# generate distributions of the annotated dataset
def generate_annotated_dataset_dists():
    uid_list = []
    pos_list = []
    neg_list = []
    hate_label_list = []
    offensive_label_list = []
    toxic_label_list = []
    hot_label_list = []
    uid = 0

    label_map = {True: 'pos', False: 'neg'}
    with open('hot_speech/labeled_hot_data_202108.json', 'r') as fin:
        for line in fin:
            comment_json = json.loads(line.rstrip())
            toxicity_score = comment_json['toxicity']
            hate_label = label_map[get_majority_vote([x[0] for x in comment_json['composite_hate']])]
            offensive_label = label_map[get_majority_vote([x[0] for x in comment_json['composite_offensive']])]
            toxic_label = label_map[get_majority_vote([x[0] for x in comment_json['composite_toxic']])]
            hot_label = label_map[get_majority_vote([(x[0] | y[0] | z[0]) for x, y, z 
                                                     in zip(comment_json['composite_hate'], comment_json['composite_offensive'], comment_json['composite_toxic'])])]

            uid += 1
            uid_list.append(uid)
            pos_list.append(toxicity_score)
            neg_list.append(1-toxicity_score)
            hate_label_list.append(hate_label)
            offensive_label_list.append(offensive_label)
            toxic_label_list.append(toxic_label)
            hot_label_list.append(hot_label)

    all_labels = ['pos', 'neg']
    cached_dists = {}
    for metric in ['hate', 'offensive', 'toxic', 'hot']:
        df = pd.DataFrame.from_dict({'uid': uid_list, 'pos': pos_list, 'neg': neg_list, 'gt_label': eval(f'{metric}_label_list')})
        dataset = Dataset(df=df, labels=all_labels)
        calibration_curve = dataset.generate_calibration_curve(method='platt scaling')
        class_conditional_densities = dataset.infer_class_conditional_densities()

        cached_dists[f'{metric}_calibration_curve'] = calibration_curve
        cached_dists[f'{metric}_class_conditional_densities'] = class_conditional_densities
        
    pickle.dump(cached_dists, open('hot_speech/hot_cached_dists.pkl', 'wb'))


def get_hot_values(toxicity_values):
    # build a dataset object from the list
    num_items = len(toxicity_values)
    df = pd.DataFrame.from_dict({'uid': list(range(num_items)), 'pos': toxicity_values, 'neg': 1-np.array(toxicity_values)})
    dataset = Dataset(df=df, labels=['pos', 'neg'])

    # load the calibration curve of the hot speech dataset
    cached_dists_filepath = 'hot_speech/hot_cached_dists.pkl'
    if not os.path.exists(cached_dists_filepath):
        generate_annotated_dataset_dists()
    cached_dists = pickle.load(open(cached_dists_filepath, 'rb'))

    result = {}
    for metric in ['hate', 'offensive', 'toxic', 'hot']:
        calibration_curve = cached_dists[f'{metric}_calibration_curve']
        class_conditional_densities = cached_dists[f'{metric}_class_conditional_densities']

        # print('for metric:', metric)

        ex_prevalence_est = dataset.extrinsic_estimate(calibration_curve=calibration_curve)
        # print(f'extrinsic estimate: {ex_prevalence_est:.4f} on the a simulated data')
        result[metric] = ex_prevalence_est

        # in_prevalence_est = dataset.instrinsic_estimate(class_conditional_densities=class_conditional_densities)
        # print(f'instrinsic estimate: {in_prevalence_est:.4f} on the a simulated data')

    return result


if __name__ == '__main__':
    cx_list = np.random.rand(10000).tolist()
    print(get_hot_values(cx_list))
