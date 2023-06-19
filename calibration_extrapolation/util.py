from matplotlib import pyplot as plt
import os
from calibration_extrapolation.conf import *


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


def save_to_img(img_name):
    if img_name:
        plt.savefig(os.path.join(img_folder, f'{img_name}.{img_kind}'),
                    bbox_inches='tight')


def shift_axis(x_axis):
    return (x_axis[: -1] + x_axis[1:]) / 2


def get_bin_idx(score, size=10):
    return min(int(score * size), size-1)


def prepare_canvas():
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major')
    return ax


def one_gradient_plot(ax, x_axis, top_axis, bottom_axis=None, **kwds):
    color = kwds.get('color', unknown_color)
    label = kwds.get('label', '')

    num_bin = len(x_axis)
    bin_width = 1 / num_bin
    bin_margin = bin_width / 2

    if bottom_axis is None:
        bottom_axis = np.zeros(num_bin)

    for x, top_coord, bottom_coord in zip(x_axis, top_axis, bottom_axis):
        left_coord = x - bin_margin
        right_coord = x + bin_margin

        ax.fill_between([left_coord, right_coord],
                        [top_coord, top_coord],
                        [bottom_coord, bottom_coord],
                        facecolor=color, alpha=x, lw=0)

    ax.plot(x_axis, top_axis, label=label, c=color, lw=2, zorder=40)


def two_gradient_plot(ax, x_axis, split_axis, color_top, color_bottom):
    num_bin = len(x_axis)
    bin_width = 1 / num_bin
    bin_margin = bin_width / 2

    for x, split_coord in zip(x_axis, split_axis):
        left_coord = x - bin_margin
        right_coord = x + bin_margin

        ax.fill_between([left_coord, right_coord],
                        [0, 0],
                        [split_coord, split_coord],
                        facecolor=color_bottom, alpha=x, lw=0)

        ax.fill_between([left_coord, right_coord],
                        [split_coord, split_coord],
                        [1, 1],
                        facecolor=color_top, alpha=x, lw=0)

    ax.plot(x_axis, split_axis, c='k', lw=2, zorder=40)



def plot_stacked_frequency(x_axis, freq_hist, calibration_curve, ax=None, fig_name=None):
    if ax is None:
        ax = prepare_canvas()

    cali_prob_array = calibration_curve.get_calibrated_prob(x_axis)
    weighted_freq_hist = cali_prob_array * freq_hist

    one_gradient_plot(ax, x_axis, weighted_freq_hist, color=positive_color)
    one_gradient_plot(ax, x_axis, freq_hist, bottom_axis=weighted_freq_hist,
                      color=negative_color)

    ax.set_xlabel('$C(X)$')
    ax.set_ylabel('freq')
    ax.set_ylim(ymin=0)

    if fig_name:
        plt.savefig(f'{fig_name}.svg', bbox_inches='tight')




