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
from info import MATERIALS, MEMORY
from plots import plot_spectra_2conditions
from settings import COLORS, FREQ_RANGE

# settings
plt.style.use('mplstyle/default.mplstyle')
FIGSIZE = [5, 2]


def main():

    # load results of step 3
    fname = f"{PROJECT_PATH}/data/results/ieeg_modulated_channels.csv"
    stats = pd.read_csv(fname, index_col=0)

    # make directory for output figures
    dir_fig = f"{PROJECT_PATH}/figures/group_spectra"
    if not os.path.exists(f"{dir_fig}"): 
        os.makedirs(f"{dir_fig}")

    for feature in ['alpha', 'gamma', 'all', 'any']:
        plot_group_spectra(stats, feature, dir_fig)


def plot_group_spectra(stats, feature, dir):

    # create figure
    _, axes = plt.subplots(1, 2, figsize=FIGSIZE)

    for ax, material, color in zip(axes, MATERIALS, ['brown', 'blue']):
        # load data
        fname = f"{PROJECT_PATH}/data/ieeg_spectral_results/psd_{material}_hit_XXXstim.npz"
        data_pre = np.load(fname.replace("XXX", "pre"))
        data_post = np.load(fname.replace("XXX", "post"))
        psd_pre = data_pre['spectra'][stats[f"sig_{feature}"]]
        psd_post = data_post['spectra'][stats[f"sig_{feature}"]]
        freq = data_pre['freq']

        # plot
        title = f"{material[0].upper()}{material[1:-1]}-encoding"
        colors = [COLORS[f'light_{color}'], COLORS[color]]
        f_mask = np.logical_and(freq>FREQ_RANGE[0], freq<FREQ_RANGE[1])
        plot_spectra_2conditions(psd_pre[:, f_mask], psd_post[:, f_mask], 
                                freq[f_mask], shade_sem=True, title=title, 
                                ax=ax, color=colors)
        
    # save
    plt.savefig(f"{dir}/mean_spectra_sig_{feature}.png")
    plt.close()

        
if __name__ == "__main__":
    main()
