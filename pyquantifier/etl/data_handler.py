import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot as plt

from pyquantifier.util import get_bin_idx
from pyquantifier.plot import prepare_canvas
from pyquantifier.distribution.calibration_curve import PerfectCC, \
    InferredCalibrationCurve, InferredPlattScaling


class DataHandler:
    """
    A data handler.
    """

    def __init__(self, filepath: str, ):
        self.filepath = filepath
        self._oracle_df = pd.read_csv(self.filepath, sep=',')
        self.size = self._oracle_df.shape[0]

        self.observed_df = self._oracle_df[['C(X)', 'GT']].copy()
        self.observed_df['GT'] = ''

        self._p_gt = self._oracle_df[self._oracle_df['GT'] == True].shape[0] / self.size

        self.sampled_df = None

        self.positive_color = '#3d85c6'
        self.negative_color = '#cc0000'
        self.unknown_color = '#666666'

    # some getter
    def get_observed_df(self):
        return self.observed_df

    def get_sampled_df(self):
        return self.sampled_df

    def get_oracle_prevalence(self):
        return np.mean(self._oracle_df['GT'] == True)

    def load_features(self, features):
        self.observed_df[features] = self._oracle_df[features]

    def get_oracle_labels(self, rows=[1]):
        self.observed_df.loc[rows, 'GT'] = self._oracle_df.loc[rows, 'GT']
        self.sample_df = self.observed_df[self.observed_df['GT'] != '']

    def get_sample_for_labeling(self, n_item=100, num_bin=10, strategy='random'):
        unlabeled_subset = self.observed_df[self.observed_df['GT'] == '']
        # unlabeled_subset = unlabeled_subset.sample(frac=1)
        if strategy == 'random':
            return unlabeled_subset.sample(n=n_item).index
        else:
            if strategy == 'uniform on C(X)':
                num_sample_in_bin = n_item // num_bin
                to_fill_list = [num_sample_in_bin] * num_bin
            elif strategy == 'neyman':
                strata_list, _ = np.histogram(self.observed_df['C(X)'].values, bins=np.linspace(0, 1, num_bin + 1))
                N = np.array(strata_list)
                K = np.arange(0.05, 1, 0.1)
                S = np.sqrt(K * (1 - K))
                to_fill_list = list(map(int, n_item * (N * S) / sum(N * S)))

            sampled_idx = []
            for idx, item in unlabeled_subset.iterrows():
                cx = item['C(X)']
                bin_idx = get_bin_idx(cx, num_bin)
                if to_fill_list[bin_idx] > 0:
                    sampled_idx.append(idx)
                    to_fill_list[bin_idx] -= 1

                    if sum(to_fill_list) == 0:
                        break
            return sampled_idx

    def get_labeled_sample(self):
        return self.observed_df[self.observed_df['GT'] != '']

    def get_all_oracle(self):
        self.observed_df['GT'] = self._oracle_df['GT']

    def hide_all_oracle(self):
        self.observed_df['GT'] = ''

    def count_gt(self):
        return (self.observed_df['GT'].values != '').sum()

    def get_prev(self, df):
        return df[df['GT'] == True].shape[0] / df.shape[0]

    def _prepare_canvas(self):
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=14)
        return ax

    def _plot_gradient_hist(self, top_arr, num_bin, ax, norm=False,
                            bottom_arr=None, color='k', color2='k', relation='independent',
                            data_format='hist'):
        x_axis = np.linspace(0, 1, num_bin + 1)
        if data_format == 'line':
            relative_hist = top_arr
        else:
            top_hist_freq, _ = np.histogram(top_arr, bins=x_axis, density=norm)
            if bottom_arr is None:
                bottom_hist_freq = np.zeros(num_bin + 1)
            else:
                bottom_hist_freq, _ = np.histogram(bottom_arr, bins=x_axis, density=norm)
                if relation == 'relative':
                    relative_hist = top_hist_freq / bottom_hist_freq

        for bin_idx in range(num_bin):
            left_point = x_axis[bin_idx]
            right_point = x_axis[bin_idx + 1]
            transparency = (left_point + right_point) / 2

            if relation == 'relative' or data_format == 'line':
                ax.fill_between([left_point, right_point],
                                [0, 0],
                                [relative_hist[bin_idx], relative_hist[bin_idx]],
                                facecolor=color, alpha=transparency, lw=0)
                ax.fill_between([left_point, right_point],
                                [relative_hist[bin_idx], relative_hist[bin_idx]],
                                [1, 1],
                                facecolor=color2, alpha=transparency, lw=0)
            else:
                ax.fill_between([left_point, right_point],
                                [bottom_hist_freq[bin_idx], bottom_hist_freq[bin_idx]],
                                [top_hist_freq[bin_idx], top_hist_freq[bin_idx]],
                                facecolor=color, alpha=transparency, lw=0)

        if relation == 'relative' or data_format == 'line':
            ax.plot((x_axis[1:] + x_axis[:-1]) / 2,
                    relative_hist, c='k', lw=2, zorder=50)
        else:
            ax.plot((x_axis[1:] + x_axis[:-1]) / 2,
                    top_hist_freq, c=color, lw=2, zorder=50)

    def plot_dist_classifier_scores(self, df=None, num_bin=100, fig_name=None, ax=None):
        if ax is None:
            ax = prepare_canvas()

        if df is None:
            df = self.observed_df

        self._plot_gradient_hist(df['C(X)'].values,
                                 num_bin=num_bin, ax=ax, color=self.unknown_color)

        ax.set_xlabel('$C(X)$')
        ax.set_ylabel('freq')
        ax.set_ylim(ymin=0)

        if fig_name:
            plt.savefig(f'{fig_name}.svg', bbox_inches='tight')

    def plot_stacked_frequency(self, df=None, num_bin=100, fig_name=None, ax=None):
        if ax is None:
            ax = prepare_canvas()

        if df is None:
            df = self.observed_df

        self._plot_gradient_hist(df[df['GT'] == True]['C(X)'].values,
                                 num_bin=num_bin, ax=ax, color=self.positive_color)
        self._plot_gradient_hist(df['C(X)'].values,
                                 bottom_arr=df[df['GT'] == True]['C(X)'].values,
                                 num_bin=num_bin, ax=ax, color=self.negative_color)

        ax.set_xlabel('$C(X)$')
        ax.set_ylabel('freq')
        ax.set_ylim(ymin=0)

        if fig_name:
            plt.savefig(f'{fig_name}.svg', bbox_inches='tight')

    def plot_class_conditional_density(self, df=None, num_bin=100, fig_name=None, ax=None):
        if ax is None:
            ax = prepare_canvas()

        if df is None:
            df = self.observed_df

        self._plot_gradient_hist(df[df['GT'] == True]['C(X)'].values, norm=True,
                                 num_bin=num_bin, ax=ax, color=self.positive_color)
        self._plot_gradient_hist(df[df['GT'] == False]['C(X)'].values, norm=True,
                                 num_bin=num_bin, ax=ax, color=self.negative_color)

        ax.set_xlabel('$C(X)$')
        ax.set_ylabel('density')
        ax.set_ylim(ymin=0)
        ax.set_yticks([])

        if fig_name:
            plt.savefig(f'{fig_name}.svg', bbox_inches='tight')

    def get_calibration_curve(self, df=None, method='perfect calibration', num_bin=100):
        if method == 'perfect calibration':
            return PerfectCC()
        elif method == 'nonparametric binning':
            return InferredCalibrationCurve(df, num_bin)
        elif method == 'platt scaling':
            return InferredPlattScaling().fit(df)

    def plot_calibration_curve(self, df=None, num_bin=100, show_diagonal=False,
                               method='perfect calibration', fig_name=None, ax=None):
        if ax is None:
            ax = prepare_canvas()

        if df is None:
            df = self.observed_df

        x_axis = np.linspace(0, 1, num_bin + 1)

        if method == 'perfect calibration':
            prob_cali_curve = np.linspace(0, 1, num_bin)
            self._plot_gradient_hist(prob_cali_curve,
                                     num_bin=num_bin, ax=ax, color=self.positive_color,
                                     color2=self.negative_color,
                                     data_format='line')
        elif method == 'platt scaling':
            df['GT'] = df['GT'].astype('bool')
            train_CX = df['C(X)'].values.reshape(-1, 1)
            train_GT = df['GT'].values
            prob_cali_func = LogisticRegression(solver='lbfgs', fit_intercept=True).fit(train_CX, train_GT)
            prob_cali_curve = prob_cali_func.predict_proba(x_axis.reshape(-1, 1))[:, -1]

            self._plot_gradient_hist(prob_cali_curve,
                                     num_bin=num_bin, ax=ax, color=self.positive_color,
                                     color2=self.negative_color,
                                     data_format='line')

        if show_diagonal:
            ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlabel('$C(X)$')
        ax.set_ylabel('$P(GT=1|C(X))$')
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylim(ymin=0)

        if fig_name:
            plt.savefig(f'{fig_name}.svg', bbox_inches='tight')

    def plot_dist_gt_labels(self, df=None, set_pp=False, p_p=1, fig_name=None, ax=None):
        if ax is None:
            ax = prepare_canvas()

        if set_pp:
            p_n = 1 - p_p
        else:
            if df is None:
                df = self.observed_df

            p_p = self.get_prev(df)
            p_n = 1 - p_p

        ax.bar([0, 1], [p_n, p_p], width=0.7,
               color=[self.negative_color, self.positive_color], lw=2, edgecolor='k')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['GT=0', 'GT=1'])
        ax.set_ylabel('$P(GT)$')
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylim([0, 1])

        ax.text(0, p_n + 0.01, f'{p_n:.2f}', color='k', ha='center', va='bottom', fontsize=14)
        ax.text(1, p_p + 0.01, f'{p_p:.2f}', color='k', ha='center', va='bottom', fontsize=14)

        if fig_name:
            plt.savefig(f'{fig_name}.svg', bbox_inches='tight')

    def generate_all_distribution_plots(self, df, num_bin=100, calibration_method='nonparametric binning'):
        fig, axes = plt.subplots(1, 5, figsize=(16, 3))
        axes = axes.ravel()

        self.plot_dist_classifier_scores(df, num_bin=num_bin, ax=axes[0])
        if calibration_method == 'nonparametric binning':
            base_calibration_curve = InferredCalibrationCurve(df, num_bin=num_bin)
        elif calibration_method == 'platt scaling':
            base_calibration_curve = InferredPlattScaling(df, num_bin=num_bin)
        base_calibration_curve.plot(ax=axes[1])
        self.plot_stacked_frequency(df, num_bin=num_bin, ax=axes[2])
        self.plot_dist_gt_labels(df, ax=axes[3])
        self.plot_class_conditional_density(df, num_bin=num_bin, ax=axes[4])

        for ax in axes:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='both', which='major')

        plt.tight_layout()
