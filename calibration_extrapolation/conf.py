import numpy as np
from matplotlib import pyplot as plt

data_folder = '../data'
img_folder = '../img'
img_kind = 'png'

num_fine_bin = 100
fine_axis = np.linspace(0, 1, num_fine_bin + 1)
num_coarse_bin = 10
coarse_axis = np.linspace(0, 1, num_coarse_bin + 1)

positive_color = '#00539C'
negative_color = '#FFD662'
unknown_color = '#333333'

plt.rc('axes', labelsize=16)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the x tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the y tick labels
plt.rc('legend', fontsize=14)    # fontsize of the legend
