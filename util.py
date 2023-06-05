from collections.abc import Iterable
import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels

positive_color = '#3d85c6'
negative_color = '#cc0000'
unknown_color = '#666666'


class CalibrationCurve():
    """
    A calibration curve.
    """
    def __init__(self):
        self.x_axis = None
        self.y_axis = None
    
    def get_calibrated_prob(self):
        pass

    def plot(self, pos_color='#3d85c6', neg_color='#cc0000', show_diagonal=False, fig_name=False, ax=None):
        if ax is None:
            ax = _prepare_canvas()

        bin_width = 1 / len(self.x_axis)
        bin_margin = bin_width / 2
        for x, y in zip(self.x_axis, self.y_axis):
            left_point = x - bin_margin
            right_point = x + bin_margin

            ax.fill_between([left_point, right_point], 
                            [0, 0], 
                            [y, y], 
                            facecolor=pos_color, alpha=x, lw=0)
            ax.fill_between([left_point, right_point], 
                            [y, y], 
                            [1, 1], 
                            facecolor=neg_color, alpha=x, lw=0)

        ax.plot(self.x_axis, self.y_axis, 'k-', lw=2)

        if show_diagonal:
            ax.plot([0, 1], [0, 1], 'k--', lw=2)

        ax.set_xlabel('$C(X)$')
        ax.set_ylabel('$P(GT=1|C(X))$')
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylim(ymin=0)

        if fig_name:
            plt.savefig(f'{fig_name}.svg', bbox_inches='tight')


def get_bin_idx(score, size=10):
    return min(int(score * size), size-1)


def _prepare_canvas():
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major')
    return ax


def _gradient_plot(x_axis, top, color, bottom=None, ratio=False, ax=None):
    if ax is None:
        ax = _prepare_canvas()

    num_bin = len(x_axis)
    bin_width = 1 / num_bin
    bin_margin = bin_width / 2

    if bottom is None:
        bottom = np.zeros(num_bin + 1)
    
    if ratio:
        for x, split_coord in zip(x_axis, top):
            left_point = x - bin_margin
            right_point = x + bin_margin

            ax.fill_between([left_point, right_point], 
                            [0, 0], 
                            [split_coord, split_coord], 
                            facecolor=positive_color, alpha=x, lw=0)
            ax.fill_between([left_point, right_point], 
                            [1, 1], 
                            [split_coord, split_coord], 
                            facecolor=positive_color, alpha=x, lw=0)

        ax.plot(x_axis, top, c=color, lw=2, zorder=50)
    else:
        for x, top_coord, bottom_coord in zip(x_axis, top, bottom):
            left_point = x - bin_margin
            right_point = x + bin_margin

            ax.fill_between([left_point, right_point], 
                            [top_coord, top_coord], 
                            [bottom_coord, bottom_coord], 
                            facecolor=color, alpha=x, lw=0)

        ax.plot(x_axis, top, c=color, lw=2, zorder=50)


def plot_stacked_frequency(x_axis, freq_hist, calibration_curve: CalibrationCurve, ax=None, fig_name=None):
    if ax is None:
        ax = _prepare_canvas()

    cali_prob_array = calibration_curve.get_calibrated_prob(x_axis)
    weighted_freq_hist = cali_prob_array * freq_hist

    _gradient_plot(x_axis, weighted_freq_hist, color=positive_color, ax=ax)
    _gradient_plot(x_axis, freq_hist, bottom=weighted_freq_hist, color=negative_color, ax=ax)       

    ax.set_xlabel('$C(X)$')
    ax.set_ylabel('freq')
    ax.set_ylim(ymin=0)

    if fig_name:
        plt.savefig(f'{fig_name}.svg', bbox_inches='tight')


class MixtureModel(stats.rv_continuous):
    """
    Generate a mixture of distributions.
    """

    def __init__(self, submodels, weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submodels = submodels
        self.num_model = len(submodels)
        weight_sum = sum(weights)
        self.weights = [weight/weight_sum for weight in weights]
        self.num_theor_slice = 100
        self.theor_cx_axis = np.linspace(0, 1, self.num_theor_slice + 1)
        self.num_empir_bin = 10
        self.empir_cx_axis = np.linspace(0, 1, self.num_empir_bin + 1)

    def _pdf(self, x):
        pdf = self.weights[0] * self.submodels[0].pdf(x)
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            pdf += weight * submodel.pdf(x)
        return pdf

    def rvs(self, size):
        submodel_choices = np.random.choice(range(self.num_model), size=size, p=self.weights)
        submodel_samples = [submodel.rvs(size=size) for submodel in self.submodels]
        rvs = np.choose(submodel_choices, submodel_samples)
        return rvs
    
    def plot_pdf_and_hist(self, size, color='k'):
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes = axes.ravel()
        
        rv_pdf = self.pdf(self.theor_cx_axis)
        for slice_idx in range(self.num_theor_slice):
            transpancy = (self.theor_cx_axis[slice_idx] + self.theor_cx_axis[slice_idx+1]) / 2
            axes[0].fill_between([self.theor_cx_axis[slice_idx], self.theor_cx_axis[slice_idx+1]], 
                                 [0, 0], 
                                 [rv_pdf[slice_idx], rv_pdf[slice_idx+1]], 
                                 facecolor=color, alpha=transpancy, lw=0)
        
        axes[0].plot(self.theor_cx_axis, rv_pdf, c=color, lw=2, zorder=50)
        axes[0].set_ylabel('$P(C(X))$', fontsize=16)

        rv_scores = self.rvs(size)
        hist, _ = np.histogram(rv_scores, bins=self.empir_cx_axis)
                
        for bin_idx in range(self.num_empir_bin):            
            transpancy = (self.empir_cx_axis[bin_idx] + self.empir_cx_axis[bin_idx+1]) / 2
            axes[1].fill_between([self.empir_cx_axis[bin_idx], self.empir_cx_axis[bin_idx+1]], 
                                 [0, 0], 
                                 [hist[bin_idx], hist[bin_idx]], 
                                 facecolor=color, alpha=transpancy, lw=0)

        axes[1].plot((self.empir_cx_axis[1:] + self.empir_cx_axis[:-1]) / 2, 
                     hist, c=color, lw=2, zorder=50)
        axes[1].set_ylabel('frequency', fontsize=16)
            
        for ax in axes:
            ax.set_xlabel('$C(X)$', fontsize=16)
            ax.set_xlim([-0.02, 1.02])
            ax.set_xticks([0, 0.5, 1])
            ax.set_ylim(ymin=0)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='both', which='major')
        
        plt.tight_layout()


class DataHandler():
    """
    A data handler.
    """

    def __init__(self, filepath: str,):
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
        elif strategy == 'uniform on C(X)':
            num_sample_in_bin = n_item // num_bin
            to_fill_list = [num_sample_in_bin] * num_bin
            sampled_idx = []
            for idx, item in unlabeled_subset.iterrows():
                cx = item['C(X)']
                bin_idx =get_bin_idx(cx, num_bin)
                if to_fill_list[bin_idx] > 0:
                    sampled_idx.append(idx)
                    to_fill_list[bin_idx] -= 1

                    if sum(to_fill_list) == 0:
                        break 
            return sampled_idx
        elif strategy == 'neyman':
            strata_list, _ = np.histogram(self.observed_df['C(X)'].values, bins=np.linspace(0, 1, num_bin+1))
            N = np.array(strata_list)
            K = np.arange(0.05, 1, 0.1)
            S = np.sqrt(K * (1 - K))
            to_fill_list = list(map(int, n_item * (N * S) / sum(N * S)))
            sampled_idx = []
            for idx, item in unlabeled_subset.iterrows():
                cx = item['C(X)']
                bin_idx =get_bin_idx(cx, num_bin)
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
            ax = _prepare_canvas()

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
            ax = _prepare_canvas()

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
            ax = _prepare_canvas()

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

    def plot_calibration_curve(self, df=None, num_bin=100, show_diagonal=False,
                               method='perfect calibration', fig_name=None, ax=None):
        if ax is None:
            ax = _prepare_canvas()

        if df is None:
            df = self.observed_df

        x_axis = np.linspace(0, 1, num_bin + 1)

        if method=='perfect calibration':
            prob_cali_curve = np.linspace(0, 1, num_bin)
            self._plot_gradient_hist(prob_cali_curve, 
                                     num_bin=num_bin, ax=ax, color=self.positive_color,
                                     color2=self.negative_color,
                                     data_format='line')
        elif method=='platt scaling':
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
            ax = _prepare_canvas()

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
            base_calibration_curve = NPBinningCalibrationCurve(df, num_bin=num_bin)
        elif calibration_method == 'platt scaling':
            base_calibration_curve = LogisticCalibrationCurve(df, num_bin=num_bin)
        base_calibration_curve.plot(ax=axes[1])
        self.plot_stacked_frequency(df, num_bin=num_bin, ax=axes[2])
        self.plot_dist_gt_labels(df, ax=axes[3])
        self.plot_class_conditional_density(df, num_bin=num_bin, ax=axes[4])

        for ax in axes:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='both', which='major')

        plt.tight_layout()


class PerfectCalibrationCurve(CalibrationCurve):
    """
    A perfect calibration curve.
    """
    def __init__(self):
        self.x_axis = np.linspace(0, 1, 200)
        self.y_axis = self.x_axis

    def get_calibrated_prob(self, cxs):
        return cxs


class NPBinningCalibrationCurve(CalibrationCurve):
    """
    A nonparametric binning calibration curve.
    """
    def __init__(self, df, num_bin):
        self.num_bin = num_bin
        self.x_axis = np.linspace(0, 1, num_bin + 1)

        pos_cx = df[df['GT'] == True]['C(X)'].values
        all_cx = df['C(X)'].values

        pos_hist_freq, _ = np.histogram(pos_cx, bins=self.x_axis)
        all_hist_freq, _ = np.histogram(all_cx, bins=self.x_axis)
        self.y_axis = pos_hist_freq / all_hist_freq

        bin_width = 1 / num_bin
        bin_margin = bin_width / 2
        self.x_axis = self.x_axis[:-1] + bin_margin

    
    def get_calibrated_prob(self, cxs):
        if isinstance(cxs, Iterable):
            return np.array([self.y_axis[get_bin_idx(cx, self.num_bin)] for cx in cxs])
        else:
            return self._find_cali_prob(cxs)


class LogisticCalibrationCurve(CalibrationCurve):
    """
    A logistic calibration curve.
    """
    def __init__(self):
        self.lr_regressor = LogisticRegression()

    def fit(self, df):
        train_CX = df['C(X)'].values.reshape(-1, 1)
        train_GT = df['GT'].astype('bool').values
        self.lr_regressor = LogisticRegression(solver='lbfgs', fit_intercept=True).fit(train_CX, train_GT)
        self.x_axis = np.linspace(0, 1, 101)
        self.y_axis = self.get_calibrated_prob(self.x_axis)

        
    def sef_params(self, w, b):
        self.lr_regressor.coef_ = np.array([[w]])
        self.lr_regressor.intercept_ = np.array([b])
        self.lr_regressor.classes_=np.array([0, 1])
    
    def get_calibrated_prob(self, cxs):
        return self.lr_regressor.predict_proba(cxs.reshape(-1, 1))[:, 1]


class ProbabilityEstimator():
    """
    A class for probability estimator.
    """
    def __init__(self):
        self.calibration_curve = None
    
    def set_calibration_curve(self, calibration_curve: CalibrationCurve):
        self.calibration_curve = calibration_curve

    def estimate(self, cx_array):
        calibrated_prob_array = self.calibration_curve.get_calibrated_prob(cx_array)
        return np.mean(calibrated_prob_array)

    def plot(self, cx_array, num_bin=100):
        x_axis = np.linspace(0, 1, num_bin + 1)
        freq_hist, _ = np.histogram(cx_array, bins=x_axis)

        num_bin = len(x_axis)
        bin_width = 1 / num_bin
        bin_margin = bin_width / 2

        x_axis = x_axis[:-1] + bin_margin
        plot_stacked_frequency(x_axis, freq_hist, self.calibration_curve, ax=None, fig_name=None)


class MixtureModelEstimator():
    """
    A class for mixture model estimator.
    """
    def __init__(self):
        self.positivity_density = None
        self.negativity_density = None
        
    def set_positive_density(self, positivity_density):
        self.positivity_density = positivity_density

    def set_negativity_density(self, negativity_density):
        self.negativity_density = negativity_density

    def fit(self, sample_df, base_cx, num_bin=10):
        x_axis = np.linspace(0, 1, num_bin+1)
        base_cx_hist, _ = np.histogram(base_cx, bins=x_axis, density=True)
        sample_cx_hist, _ = np.histogram(sample_df['C(X)'].values, bins=x_axis, density=True)

        weight = base_cx_hist / sample_cx_hist

        pos_cx = sample_df[sample_df['GT'] == True]['C(X)'].values
        neg_cx = sample_df[sample_df['GT'] == False]['C(X)'].values

        pos_hist_freq, _ = np.histogram(pos_cx, bins=x_axis, density=True)
        neg_hist_freq, _ = np.histogram(neg_cx, bins=x_axis, density=True)

        pos_hist_freq *= weight
        pos_total = np.sum(pos_hist_freq)
        pos_hist_freq /= pos_total

        neg_hist_freq *= weight
        neg_total = np.sum(neg_hist_freq)
        neg_hist_freq /= neg_total

        self.positivity_density = pos_hist_freq
        self.negativity_density = neg_hist_freq
    
    def hellinger(self, p, q):
        return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)

    def estimate(self, cx_array):
        num_bin = len(self.positivity_density)
        x_axis = np.linspace(0, 1, num_bin + 1)
        cx_hist, _ = np.histogram(cx_array, bins=x_axis, density=True)

        min_dist = 10000
        best_p_p = 0

        for p_p in np.linspace(0, 1, 101):
            dist = self.hellinger(cx_hist, self.positivity_density * p_p + self.negativity_density * (1 - p_p))
            if dist < min_dist:
                min_dist = dist
                best_p_p = p_p
        return best_p_p

    def plot(self, cx_array, num_bin=100):
        x_axis = np.linspace(0, 1, num_bin + 1)
        freq_hist, _ = np.histogram(cx_array, bins=x_axis)

        num_bin = len(x_axis)
        bin_width = 1 / num_bin
        bin_margin = bin_width / 2

        x_axis = x_axis[:-1] + bin_margin
        plot_stacked_frequency(x_axis, freq_hist, self.calibration_curve, ax=None, fig_name=None)