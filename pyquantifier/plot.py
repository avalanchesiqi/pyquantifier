import os
import numpy as np
from matplotlib import pyplot as plt

from pyquantifier.conf import *


num_fine_bin = 1000
fine_axis = np.linspace(0, 1, num_fine_bin + 1)
num_coarse_bin = 10
coarse_axis = np.linspace(0, 1, num_coarse_bin + 1)

plt.rc('axes', labelsize=16)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the x tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the y tick labels
plt.rc('legend', fontsize=14)    # fontsize of the legend

# class ColorPalette:
#     CC2 = ['#FFD662', '#00539C']
#     CC3 = ['#f6511d', '#ffb400', '#00a6ed']
#     CC4 = ['#4486F4', '#1CA45C', '#FF9E0F', '#DA483B']
#     unknown_color = '#333333'
#     pos_color = '#00539C'
#     neg_color = '#FFD662'

ColorPalette = {'pos': '#00539C',
                'neg': '#FFD662',
                'unknown': '#333333'}


ColorPalette = {'pos': '#cc0000',
                'neg': '#c9daf8',
                'unknown': '#333333'}


def save_to_img(img_name):
    if img_name:
        plt.savefig(os.path.join(img_folder, img_name),
                    bbox_inches='tight')


def shift_axis(x_axis):
    return (x_axis[: -1] + x_axis[1:]) / 2


def prepare_canvas():
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major')
    return ax


def one_gradient_plot(ax, x_axis, top_axis, bottom_axis=None, **kwds):
    color = kwds.get('color', ColorPalette['unknown'])
    label = kwds.get('label', '')
    edge = kwds.get('edge', True)
    edge_color = kwds.get('edge_color', color)

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

    if edge:
        ax.plot(x_axis, top_axis, label=label, c=edge_color, lw=2, zorder=40)


# def two_gradient_plot(ax, x_axis, split_axis, color_top, color_bottom):
#     num_bin = len(x_axis)
#     bin_width = 1 / num_bin
#     bin_margin = bin_width / 2

#     for x, split_coord in zip(x_axis, split_axis):
#         left_coord = x - bin_margin
#         right_coord = x + bin_margin

#         ax.fill_between([left_coord, right_coord],
#                         [0, 0],
#                         [split_coord, split_coord],
#                         facecolor=color_bottom, alpha=x, lw=0)

#         ax.fill_between([left_coord, right_coord],
#                         [split_coord, split_coord],
#                         [1, 1],
#                         facecolor=color_top, alpha=x, lw=0)

#     ax.plot(x_axis, split_axis, c='k', lw=2, zorder=40)


def plot_stacked_frequency(x_axis, freq_hist, calibration_curve, ax=None, fig_name=None):
    if ax is None:
        ax = prepare_canvas()

    cali_prob_array = calibration_curve.get_calibrated_prob(x_axis)
    weighted_freq_hist = cali_prob_array * freq_hist

    one_gradient_plot(ax, x_axis, weighted_freq_hist,
                      color=ColorPalette.two_color[0])
    one_gradient_plot(ax, x_axis, freq_hist, bottom_axis=weighted_freq_hist,
                      color=ColorPalette.two_color[1])

    ax.set_xlabel('$C(X)$')
    ax.set_ylabel('freq')
    ax.set_ylim(ymin=0)

    if fig_name:
        plt.savefig(fig_name, bbox_inches='tight')


def plot_empirical_hist(data, **kwds):
    """Plot the histogram for empirical data.

    Parameters
    ----------
    data : list
        List of data points
    ax : AxesSubplot, optional
        Axis to plot the pdf
    color : str, optional
        Main color for the plot
    fig_name : str, optional
        Filepath of the output figure

    """
    ax = kwds.get('ax', None)
    if ax is None:
        ax = prepare_canvas()
    color = kwds.get('color', ColorPalette['unknown'])
    density = kwds.get('density', False)
    ylabel = kwds.get('ylabel', 'frequency')
    img_name = kwds.get('img_name', None)

    hist, _ = np.histogram(data, bins=coarse_axis, density=density)
    one_gradient_plot(ax, shift_axis(coarse_axis), hist, color=color)

    ax.set_xlabel('$C(X)$')
    ax.set_ylabel(ylabel)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim(ymin=0)
    save_to_img(img_name)


def plot_empirical_bar(data, **kwds):
    """Plot the histogram for empirical data.

    Parameters
    ----------
    data : dict
        Dict of data labels
    ax : AxesSubplot, optional
        Axis to plot the pdf
    color : str, optional
        Main color for the plot
    fig_name : str, optional
        Filepath of the output figure

    """
    ax = kwds.get('ax', None)
    if ax is None:
        ax = prepare_canvas()
    img_name = kwds.get('img_name', None)

    num_label = len(data)
    x_axis = np.arange(num_label)
    label_axis = sorted(data.keys())
    density_axis = [data[k] for k in label_axis]

    # if num_label == 2:
    #     color_palette = ColorPalette.CC2

    ax.bar(x_axis, density_axis, width=0.7, color=[ColorPalette[k] for k in label_axis],
           lw=2, edgecolor='k')
    ax.set_xticks(x_axis)
    ax.set_xticklabels(label_axis)
    ax.set_ylabel('$P(GT)$')
    ax.set_yticks([0, 0.5, 1])

    for k, v in zip(x_axis, density_axis):
        ax.text(k, v + 0.01, f'{v:.2f}', color='k',
                ha='center', va='bottom', fontsize=16)

    save_to_img(img_name)
