import os, sys, json, pickle, re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from collections import Counter
import numpy as np
import pandas as pd

from pyquantifier.data import Dataset


def obj2str(datetime_obj, str_format='%Y-%m-%d %H:%M:%S'):
    return datetime_obj.strftime(str_format)


def str2obj(datetime_str, str_format='%Y-%m-%d %H:%M:%S'):
    return datetime.strptime(datetime_str, str_format)


def get_majority_vote(lst):
    return Counter(lst).most_common(1)[0][0]


def text_preprocess(text_string, is_reddit):
    """Process a text string.
    1) Skip removed, deleted, bot comments on Reddit
    2) Remove all quoted texts on Reddit
    3) Remove urls
    4) Remove @ mentions
    5) Replace multiple whitespaces with one whitespace
    6) Skip comments with only whitespaces and punctuations
    """
    if is_reddit:
        if text_string == '[removed]' or text_string == '[deleted]' \
            or 'I am a bot' in text_string or "I'm a bot" in text_string \
                or 'generated by a bot' in text_string:
            return ''
        reddit_quote_regex = r'>.*(\n)\1'
        text_string = re.sub(reddit_quote_regex, ' ', text_string)

    # find all http or https urls and replace them with a whitespace
    url_regex = r'https?://\S+'
    text_string = re.sub(url_regex, ' ', text_string)

    # find all @ mentions and replace them with a whitespace
    mention_regex = r'@[\w\-]+'
    text_string = re.sub(mention_regex, ' ', text_string)

    # replace multiple whitespaces with one whitespace and strip head and tail
    space_pattern = r'\s+'
    text_string = re.sub(space_pattern, ' ', text_string).strip()

    # check if text_string only contains whitespaces and punctuations
    # if so, return an empty string
    if re.match(r'^[^\w]+$', text_string):
        return ''
    return text_string


# generate distributions of the annotated dataset
def generate_annotated_dataset_dists():
    uid_list = []
    platform_list = []
    pos_list = []
    neg_list = []
    hate_label_list = []
    offensive_label_list = []
    toxic_label_list = []
    hot_label_list = []
    uid = 0

    label_map = {True: 'pos', False: 'neg'}
    with open('hot_speech/data/labeled_hot_data_202108.json', 'r') as fin:
        for line in fin:
            comment_json = json.loads(line.rstrip())
            platform = comment_json['platform']
            # toxicity_score = comment_json['toxicity']
            toxicity_score = comment_json['perspective_20231224']
            hate_label = label_map[get_majority_vote([x[0] for x in comment_json['composite_hate']])]
            offensive_label = label_map[get_majority_vote([x[0] for x in comment_json['composite_offensive']])]
            toxic_label = label_map[get_majority_vote([x[0] for x in comment_json['toxic_1_list']])]
            hot_label = label_map[get_majority_vote([(x[0] | y[0] | z[0]) for x, y, z 
                                                     in zip(comment_json['composite_hate'], comment_json['composite_offensive'], comment_json['composite_toxic'])])]

            uid += 1
            uid_list.append(uid)
            platform_list.append(platform)
            pos_list.append(toxicity_score)
            neg_list.append(1-toxicity_score)
            hate_label_list.append(hate_label)
            offensive_label_list.append(offensive_label)
            toxic_label_list.append(toxic_label)
            hot_label_list.append(hot_label)
    
    base_reddit_toxicity_list = []
    base_twitter_toxicity_list = []
    base_youtube_toxicity_list = []
    with open('hot_speech/data/202108_base_hot_comments.json', 'r') as fin:
        for line in fin:
            article_json = json.loads(line.rstrip())
            reported_rd_comments = article_json['reported_rd_comments']
            reported_tw_replies = article_json['reported_tw_replies']
            reported_yt_comments = article_json['reported_yt_comments']
            base_reddit_toxicity_list.extend([x['toxicity'] for x in reported_rd_comments if isinstance(x['toxicity'], float) and text_preprocess(x['text'], is_reddit=True)])
            base_twitter_toxicity_list.extend([x['toxicity'] for x in reported_tw_replies if isinstance(x['toxicity'], float) and text_preprocess(x['text'], is_reddit=False)])
            base_youtube_toxicity_list.extend([x['toxicity'] for x in reported_yt_comments if isinstance(x['toxicity'], float) and text_preprocess(x['text'], is_reddit=False)])

    all_labels = ['pos', 'neg']
    cached_dists = {}
    for metric in ['hate', 'offensive', 'toxic', 'hot']:
        df = pd.DataFrame.from_dict({'uid': uid_list, 'platform': platform_list, 'pos': pos_list, 'neg': neg_list, 'gt_label': eval(f'{metric}_label_list')})

        for platform in ['reddit', 'twitter', 'youtube']:
            # use the pyquantifier package to make estimations
            platform_df = df[df.platform == platform]

            # compute the selection weights
            base_toxicity_list = eval(f'base_{platform}_toxicity_list')
            bins = np.linspace(0, 1, 11)
            selection_weights = np.histogram(platform_df['pos'].tolist(), bins=bins)[0] / np.histogram(base_toxicity_list, bins=bins)[0]
            print(selection_weights)

            dataset = Dataset(df=platform_df, labels=all_labels)
            calibration_curve = dataset.generate_calibration_curve(method='platt scaling')
            class_conditional_densities = dataset.infer_class_conditional_densities(num_bin=10, selection_weights=selection_weights)

            cached_dists[f'{platform}_{metric}_calibration_curve'] = calibration_curve
            cached_dists[f'{platform}_{metric}_class_conditional_densities'] = class_conditional_densities
        
    pickle.dump(cached_dists, open('hot_speech/hot_cached_jds/current.pkl', 'wb'))


def single_hot_prevalence_estimate(cx_list, platform, metric, cached_dists, assumption='extrinsic'):
    # build a dataset object from the list
    num_item = len(cx_list)
    if num_item == 0:
        return 0

    df = pd.DataFrame.from_dict({'uid': list(range(num_item)), 'pos': cx_list, 'neg': 1-np.array(cx_list)})
    # use the pyquantifier package to make estimations
    dataset = Dataset(df=df, labels=['pos', 'neg'])

    if assumption == 'extrinsic':
        calibration_curve = cached_dists[f'{platform}_{metric}_calibration_curve']
        est_prevalence = dataset.extrinsic_estimate(calibration_curve=calibration_curve)
        # print(f'extrinsic estimate: {est_prevalence:.4f} on the a simulated data')
    else:
        class_conditional_densities = cached_dists[f'{platform}_{metric}_class_conditional_densities']
        est_prevalence = dataset.intrinsic_estimate(class_conditional_densities=class_conditional_densities)
        # print(f'intrinsic estimate: {est_prevalence:.4f} on the a simulated data')
    return est_prevalence * num_item


def hot_prevalence_estimate(cx_list, platform, metric, cached_dists, assumption='extrinsic', bootstrap=False):
    # build a dataset object from the list
    num_item = len(cx_list)
    if num_item == 0:
        return [0, 0, 0]
    
    if bootstrap:
        ret_list = []
        num_bootstrap = 100
        for _ in range(num_bootstrap):
            bootstrapped_cx_list = np.random.choice(cx_list, size=num_item, replace=True)
            bootstrapped_estimates = single_hot_prevalence_estimate(bootstrapped_cx_list, platform, metric, cached_dists, assumption)
            ret_list.append(bootstrapped_estimates)
        return [np.mean(ret_list), np.percentile(ret_list, 2.5), np.percentile(ret_list, 97.5)]
    else:
        ret_est = single_hot_prevalence_estimate(cx_list, platform, metric, cached_dists, assumption)
        return [ret_est, ret_est, ret_est]


# load data
def load_extrapolation_comments(filepath, start_date, end_date, toxicity_field='toxicity'):
    platform2field_dict = {'reddit': 'selected_reddit_comments',
                           'twitter': 'selected_twitter_replies',
                           'youtube': 'selected_youtube_comments'}
    platform_list = ['reddit', 'twitter', 'youtube']

    start_date = str2obj(start_date, '%Y-%m-%d')
    end_date = str2obj(end_date, '%Y-%m-%d')
    time_duration = (end_date - start_date).days + 1
    extrapolation_comment_dict = {}
    
    for time_lag in range(time_duration):
        target_date = obj2str(start_date + timedelta(days=time_lag), '%Y-%m-%d')
        extrapolation_comment_dict[target_date] = {
            'reddit_toxic_list': [],
            'twitter_toxic_list': [],
            'youtube_toxic_list': [],
        }
    
    with open(filepath, 'r') as fin:            
        for line in fin:
            url_json = json.loads(line.rstrip())
            url_published_at = url_json['url_published_at'][:10]
            for platform in platform_list:
                for comment in url_json[platform2field_dict[platform]]:
                    if platform == 'reddit':
                        is_reddit = True
                    else:
                        is_reddit = False
                    text = comment['text']
                    processed_text = text_preprocess(text, is_reddit)

                    if processed_text:
                        toxicity_score = comment[toxicity_field]
                        if isinstance(toxicity_score, float):
                            extrapolation_comment_dict[url_published_at][f'{platform}_toxic_list'].append(toxicity_score)

    return extrapolation_comment_dict


def main():
    # generate jds
    old_cached_dists_filepath = 'hot_speech/hot_cached_jds/until_2022-05-11.pkl'
    # if not os.path.exists(old_cached_dists_filepath):
    #     generate_annotated_dataset_dists()
    old_cached_dists = pickle.load(open(old_cached_dists_filepath, 'rb'))
    current_cached_dists_filepath = 'hot_speech/hot_cached_jds/current.pkl'
    # if not os.path.exists(current_cached_dists_filepath):
    #     generate_annotated_dataset_dists()
    current_cached_dists = pickle.load(open(current_cached_dists_filepath, 'rb'))

    # ----------------- #
    # load every day data
    cached_extrapolation_comment_dict_filepath = 'hot_speech/data/hot_extrapolation_comment_dict_consistent.pkl'
    if not os.path.exists(cached_extrapolation_comment_dict_filepath):
        comment_2022_dict = load_extrapolation_comments('hot_speech/data/2022_classified_hot_comments.json', start_date='2022-01-01', end_date='2022-12-31',  toxicity_field='toxicity')
        # comment_2023_dict = load_extrapolation_comments('hot_speech/2023_classified_hot_comments.json', start_date='2023-01-01', end_date='2023-06-19',  toxicity_field='toxicity')
        # extrapolation_comment_dict = {**comment_2022_dict, **comment_2023_dict}
        extrapolation_comment_dict = comment_2022_dict
        pickle.dump(extrapolation_comment_dict, open(cached_extrapolation_comment_dict_filepath, 'wb'))
    else:
        extrapolation_comment_dict = pickle.load(open(cached_extrapolation_comment_dict_filepath, 'rb'))
    # ----------------- #

    platform_list = ['reddit', 'twitter', 'youtube']
    date_list = sorted(extrapolation_comment_dict.keys())
    data_dict = {'date': date_list}
    for platform in platform_list:
        data_dict[f'{platform}_num_comment_list'] = []
        # data_dict[f'{platform}_hate_list'] = []
        # data_dict[f'{platform}_hate_ub_list'] = []
        # data_dict[f'{platform}_hate_lb_list'] = []
        # data_dict[f'{platform}_offensive_list'] = []
        # data_dict[f'{platform}_offensive_ub_list'] = []
        # data_dict[f'{platform}_offensive_lb_list'] = []
        data_dict[f'{platform}_toxic_list'] = []
        data_dict[f'{platform}_toxic_ub_list'] = []
        data_dict[f'{platform}_toxic_lb_list'] = []
        data_dict[f'{platform}_toxic_notcalib_list'] = []
        data_dict[f'{platform}_toxic_threshold_list'] = []
        data_dict[f'{platform}_hot_list'] = []
        data_dict[f'{platform}_hot_ub_list'] = []
        data_dict[f'{platform}_hot_lb_list'] = []

    for day in date_list:
        for platform in platform_list:
            platform_cx_list = extrapolation_comment_dict[day][f'{platform}_toxic_list']
            num_item = len(platform_cx_list)
            data_dict[f'{platform}_num_comment_list'].append(num_item)
            data_dict[f'{platform}_toxic_notcalib_list'].append(np.sum(platform_cx_list))
            data_dict[f'{platform}_toxic_threshold_list'].append(sum([1 for x in platform_cx_list if x >= 0.7]))
            for metric in ['toxic', 'hot']:
                if day < '2022-05-12':
                    est_mean, est_lb, est_ub = hot_prevalence_estimate(platform_cx_list, platform, metric, old_cached_dists, assumption='intrinsic', bootstrap=True)
                else:
                    est_mean, est_lb, est_ub = hot_prevalence_estimate(platform_cx_list, platform, metric, current_cached_dists, assumption='intrinsic', bootstrap=True)
                data_dict[f'{platform}_{metric}_list'].append(est_mean)
                data_dict[f'{platform}_{metric}_ub_list'].append(est_ub)
                data_dict[f'{platform}_{metric}_lb_list'].append(est_lb)
        print(f'finish processing {day}')
    
    data_df = pd.DataFrame(data_dict)
    data_df.index = data_df.date
    data_df.index = pd.to_datetime(data_df.index)
    data_df.drop(columns=['date'], inplace=True)

    data_df.to_csv('hot_speech/intrinsic_hot_prevalence_estimation_2022.csv')


if __name__ == '__main__':
    main()
