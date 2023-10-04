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


def preprocess(text_string, is_reddit):
    """Accepts a text string and replaces:
    1) lots of whitespace with one instance
    2) remove urls
    3) remove mentions
    4) remove quoted text on Reddit
    5) skip removed, deleted, bot comments on Reddit

    This allows us to get standardized counts of urls and mentions.
    Without caring about specific people mentioned.
    """
    if is_reddit:
        if text_string == '[removed]' or text_string == '[deleted]' \
                or 'I am a bot' in text_string:
            return ''
        reddit_quote_regex = r'>[\w\-\s]+(\n)\1'
        text_string = re.sub(reddit_quote_regex, ' ', text_string)

    space_pattern = r'\s+'
    text_string = re.sub(space_pattern, ' ', text_string)

    giant_url_regex = (r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       r'[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    text_string = re.sub(giant_url_regex, ' ', text_string)

    mention_regex = r'@[\w\-]+'
    text_string = re.sub(mention_regex, ' ', text_string)
    return text_string.strip()


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
    with open('hot_speech/labeled_hot_data_202108.json', 'r') as fin:
        for line in fin:
            comment_json = json.loads(line.rstrip())
            platform = comment_json['platform']
            toxicity_score = comment_json['toxicity']
            hate_label = label_map[get_majority_vote([x[0] for x in comment_json['composite_hate']])]
            offensive_label = label_map[get_majority_vote([x[0] for x in comment_json['composite_offensive']])]
            toxic_label = label_map[get_majority_vote([x[0] for x in comment_json['composite_toxic']])]
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
    
    original_reddit_toxicity_list = []
    original_twitter_toxicity_list = []
    original_youtube_toxicity_list = []
    with open('hot_speech/202108_base_hot_comments.json', 'r') as fin:
        for line in fin:
            article_json = json.loads(line.rstrip())
            reported_rd_comments = article_json['reported_rd_comments']
            reported_tw_replies = article_json['reported_tw_replies']
            reported_yt_comments = article_json['reported_yt_comments']
            original_reddit_toxicity_list.extend([x['toxicity_0216'] for x in reported_rd_comments if isinstance(x['toxicity_0216'], float)])
            original_twitter_toxicity_list.extend([x['toxicity_0216'] for x in reported_tw_replies if isinstance(x['toxicity_0216'], float)])
            original_youtube_toxicity_list.extend([x['toxicity_0216'] for x in reported_yt_comments if isinstance(x['toxicity_0216'], float)])    

    all_labels = ['pos', 'neg']
    cached_dists = {}
    for metric in ['hate', 'offensive', 'toxic', 'hot']:
        df = pd.DataFrame.from_dict({'uid': uid_list, 'platform': platform_list, 'pos': pos_list, 'neg': neg_list, 'gt_label': eval(f'{metric}_label_list')})

        for platform in ['reddit', 'twitter', 'youtube']:
            # use the pyquantifier package to make estimations
            platform_df = df[df.platform == platform]

            # compute the selection weights
            original_toxicity_list = eval(f'original_{platform}_toxicity_list')
            bins = np.linspace(0, 1, 11)
            selection_weights = np.histogram(platform_df['pos'].tolist(), bins=bins)[0] / np.histogram(original_toxicity_list, bins=bins)[0]
            print(selection_weights)

            dataset = Dataset(df=platform_df, labels=all_labels)
            calibration_curve = dataset.generate_calibration_curve(method='platt scaling')
            class_conditional_densities = dataset.infer_class_conditional_densities(num_bin=10, selection_weights=selection_weights)

            cached_dists[f'{platform}_{metric}_calibration_curve'] = calibration_curve
            cached_dists[f'{platform}_{metric}_class_conditional_densities'] = class_conditional_densities
        
    pickle.dump(cached_dists, open('hot_speech/hot_cached_dists.pkl', 'wb'))



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
        est_prevalence = dataset.instrinsic_estimate(class_conditional_densities=class_conditional_densities)
        # print(f'instrinsic estimate: {est_prevalence:.4f} on the a simulated data')
    return est_prevalence * num_item


def hot_prevalence_estimate(cx_list, platform, metric, assumption='extrinsic', bootstrap=False):
    # build a dataset object from the list
    num_item = len(cx_list)
    if num_item == 0:
        return [0, 0, 0]
    
    # load the calibration curve of the hot speech dataset
    cached_dists_filepath = 'hot_speech/hot_cached_dists.pkl'
    if not os.path.exists(cached_dists_filepath):
        generate_annotated_dataset_dists()
    cached_dists = pickle.load(open(cached_dists_filepath, 'rb'))
    
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
def load_2022_comments(filepath):
    platform2field_dict = {'reddit': 'selected_reddit_comments',
                           'twitter': 'selected_twitter_replies',
                           'youtube': 'selected_youtube_comments'}
    platform_list = ['reddit', 'twitter', 'youtube']

    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 12, 31)
    time_duration = (end_date - start_date).days + 1
    comment_2022_dict = {}
    
    for time_lag in range(time_duration):
        target_date = obj2str(start_date + timedelta(days=time_lag), '%Y-%m-%d')
        comment_2022_dict[target_date] = {
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
                    processed_text = preprocess(text, is_reddit)

                    if processed_text:
                        toxicity_score = comment['toxicity_0319']
                        if isinstance(toxicity_score, float):
                            comment_2022_dict[url_published_at][f'{platform}_toxic_list'].append(toxicity_score)

    return comment_2022_dict


def main():
    # ----------------- #
    # load every day data
    cached_comment_2022_dict_filepath = 'hot_speech/hot_comment_2022_dict.pkl'
    if not os.path.exists(cached_comment_2022_dict_filepath):
        comment_2022_dict = load_2022_comments('hot_speech/2022_classified_hot_comments.json')
        pickle.dump(comment_2022_dict, open(cached_comment_2022_dict_filepath, 'wb'))
    else:
        comment_2022_dict = pickle.load(open(cached_comment_2022_dict_filepath, 'rb'))
    # ----------------- #

    platform_list = ['reddit', 'twitter', 'youtube']
    date_list = sorted(comment_2022_dict.keys())
    data_dict = {'date': date_list}
    for platform in platform_list:
        data_dict[f'{platform}_num_comment_list'] = []
        data_dict[f'{platform}_hate_list'] = []
        data_dict[f'{platform}_hate_ub_list'] = []
        data_dict[f'{platform}_hate_lb_list'] = []
        data_dict[f'{platform}_offensive_list'] = []
        data_dict[f'{platform}_offensive_ub_list'] = []
        data_dict[f'{platform}_offensive_lb_list'] = []
        data_dict[f'{platform}_toxic_list'] = []
        data_dict[f'{platform}_toxic_ub_list'] = []
        data_dict[f'{platform}_toxic_lb_list'] = []
        data_dict[f'{platform}_hot_list'] = []
        data_dict[f'{platform}_hot_ub_list'] = []
        data_dict[f'{platform}_hot_lb_list'] = []

    for day in date_list:
        for platform in platform_list:
            platform_cx_list = comment_2022_dict[day][f'{platform}_toxic_list']
            num_item = len(platform_cx_list)
            data_dict[f'{platform}_num_comment_list'].append(num_item)
            for metric in ['hot', 'hate', 'offensive', 'toxic']:
                est_mean, est_lb, est_ub = hot_prevalence_estimate(platform_cx_list, platform, metric, assumption='extrinsic', bootstrap=True)
                data_dict[f'{platform}_{metric}_list'].append(est_mean)
                data_dict[f'{platform}_{metric}_ub_list'].append(est_ub)
                data_dict[f'{platform}_{metric}_lb_list'].append(est_lb)
        print(f'finish processing {day}')
    
    data_df = pd.DataFrame(data_dict)
    data_df.index = data_df.date
    data_df.index = pd.to_datetime(data_df.index)
    data_df.drop(columns=['date'], inplace=True)

    data_df.to_csv('hot_speech/extrinsic_hot_prevalence_estimation_2022.csv')


if __name__ == '__main__':
    main()
