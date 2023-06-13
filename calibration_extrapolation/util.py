import numpy as np
from matplotlib import pyplot as plt

plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels

positive_color = '#3d85c6'
negative_color = '#cc0000'
unknown_color = '#666666'


class DiscreteClassDensity:
    """
    discrete class density.
    """
    def __init__(self):
        pass


class ContinuousClassDensity:
    """
    continuous class density.
    """
    def __init__(self):
        pass


def get_bin_idx(score, size=10):
    return min(int(score * size), size-1)


def prepare_canvas():
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major')
    return ax


def _gradient_plot(x_axis, top, color, bottom=None, ratio=False, ax=None):
    if ax is None:
        ax = prepare_canvas()

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


def plot_stacked_frequency(x_axis, freq_hist, calibration_curve, ax=None, fig_name=None):
    if ax is None:
        ax = prepare_canvas()

    cali_prob_array = calibration_curve.get_calibrated_prob(x_axis)
    weighted_freq_hist = cali_prob_array * freq_hist

    _gradient_plot(x_axis, weighted_freq_hist, color=positive_color, ax=ax)
    _gradient_plot(x_axis, freq_hist, bottom=weighted_freq_hist, color=negative_color, ax=ax)       

    ax.set_xlabel('$C(X)$')
    ax.set_ylabel('freq')
    ax.set_ylim(ymin=0)

    if fig_name:
        plt.savefig(f'{fig_name}.svg', bbox_inches='tight')




