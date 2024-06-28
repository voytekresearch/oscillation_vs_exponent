"""
This script plots the average power spectra for channels identified in
scripts.3_id_modulated_channels.py as being task-modulated.

"""

# Imports - standard
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from info import MATERIALS
from plots import plot_spectra_2conditions, beautify_ax
from settings import COLORS, FREQ_RANGE, WIDTH

# settings
plt.style.use('mplstyle/default.mplstyle')


def main():

    # load results of step 3
    fname = f"{PROJECT_PATH}/data/results/ieeg_modulated_channels.csv"
    stats = pd.read_csv(fname, index_col=0)

    # make directory for output figures
    dir_fig = f"{PROJECT_PATH}/figures/group_spectra"
    dir_fig_ = f"{PROJECT_PATH}/figures/main_figures"
    for path in [dir_fig, dir_fig_]:
        if not os.path.exists(path): 
            os.makedirs(path)

    # plot for each feature
    for feature in ['alpha', 'gamma', 'all', 'any']:
        plot_group_spectra(stats, feature, dir_fig)

    # plot main figure
    plot_group_spectra(stats, 'all', dir_fig_)


def plot_group_spectra(stats, feature, dir):

    # create figure
    _, axes = plt.subplots(1, 2, figsize=[WIDTH['1col'], WIDTH['1col']/2])

    for ax, material, color in zip(axes, MATERIALS, ['brown', 'blue']):
        # load data
        fname = f"{PROJECT_PATH}/data/ieeg_spectral_results/psd_{material}_hit_XXXstim.npz"
        data_pre = np.load(fname.replace("XXX", "pre"))
        data_post = np.load(fname.replace("XXX", "post"))
        psd_pre = data_pre['spectra'][stats[f"sig_{feature}"]]
        psd_post = data_post['spectra'][stats[f"sig_{feature}"]]
        freq = data_pre['freq']

        # plot
        title = f"{material[0].upper()}{material[1:-1]} encoding"
        colors = [COLORS[f'light_{color}'], COLORS[color]]
        f_mask = np.logical_and(freq>FREQ_RANGE[0], freq<FREQ_RANGE[1])
        plot_spectra_2conditions(psd_pre[:, f_mask], psd_post[:, f_mask], 
                                freq[f_mask], shade_sem=True, title=title, 
                                ax=ax, color=colors)
        
        # beautify
        ax.grid(False)
        beautify_ax(ax)
        
    # save
    fname_out = f"mean_spectra_sig_{feature}"
    plt.savefig(f"{dir}/{fname_out}")
    plt.savefig(f"{dir}/{fname_out}.png")
    plt.close()

        
if __name__ == "__main__":
    main()
