import unittest
import numpy as np
from scipy.stats import beta

from pyquantifier.distributions import MixtureCUD
from pyquantifier.calibration_curve import PlattScaling
from pyquantifier.data import Item, Dataset

test_case_list = [
    (MixtureCUD(components=[beta(10, 2), beta(2, 5)], weights=[2, 8]),  # classifer_score_density_rv
     (21.92, -14.61),  # (w, b)
     1000000,  # num_base_size
     1000000,  # num_sample
     10),  # num_bin
    (MixtureCUD(components=[beta(10, 2), beta(2, 5)], weights=[2, 8]),
     (10, -5),
     1000000,
     1000000,
     10)
    ]


class TestExtrinsicGenerator(unittest.TestCase):
    def setUp(self):
        pass

    def test_extrinsic_generator(self):
        for case_num, test_case in enumerate(test_case_list):
            classifer_score_density_rv, (w, b), num_base_size, num_sample, num_bin = test_case

            # 1. Generate a dataset with `num_sample` items,
            # which pos scores follow the `classifer_score_density` distribution
            all_labels = ['pos', 'neg']
            simulated_pos_scores = classifer_score_density_rv.generate_data(num_base_size)

            calibration_curve = PlattScaling()
            calibration_curve.set_params(w, b)

            items = []
            gt_label_dict = {}
            for idx, pos_score in enumerate(simulated_pos_scores):
                item = Item(uid=idx+1, all_labels=all_labels, all_probs=[pos_score, 1-pos_score])
                items.append(item)

                calibrated_pos_score = calibration_curve.get_calibrated_prob(pos_score)[0]
                gt_label_dict[item.uid] = np.random.choice(all_labels, p=[calibrated_pos_score, 1-calibrated_pos_score])
            # print(gt_label_dict)

            dataset = Dataset(items=items)

            # 2. Select a subset of the dataset and annotate the labels
            selected_dataset, selection_weights = dataset.select_sample_for_annotation(
                n=num_sample,
                strategy='random', 
                bins=num_bin
                )
            annotated_labels = [gt_label_dict[uid] for uid in selected_dataset.df['uid'].values.astype(int)]
            selected_dataset.annotate_sample(annotated_labels)

            # 3. Model the sampled dataset and retrieve the estimated w and b
            selected_dataset.update_dataset_model(num_bin=num_bin)
            selected_dataset.update_calibration_curve(method='platt scaling')
            estimated_w, estimated_b = selected_dataset.calibration_curve.get_params()
            estimated_w = estimated_w.item()
            estimated_b = estimated_b.item()

            # 4. Check if the estimated w and b are close to the ground truth synthetic w and b
            with self.subTest(msg=f'Checking case number {case_num+1}'):
                print(f'\nestimated w: {estimated_w:.4f}')
                print(f'estimated b: {estimated_b:.4f}')
                self.assertAlmostEqual(estimated_w, 
                                       w, 
                                       delta=0.2,
                                       msg=f'Test failed: estimated w does not match expected result'
                                       )
                self.assertAlmostEqual(estimated_b, 
                                       b, 
                                       delta=0.2,
                                       msg=f'Test failed: estimated b does not match expected result'
                                       )
    
    def test_extrinsic_generator2(self):
        with self.subTest(msg=f'Checking case 2'):
            pass    



if __name__ == "__main__":
    unittest.main()
