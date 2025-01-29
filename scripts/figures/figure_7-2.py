"""
Figure 7-2

Aperiodic-corrected spectrograms
"""

# Imports - standard
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from specparam import SpectralGroupModel

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from info import MATERIALS
from utils import get_start_time, print_time_elapsed
from tfr_utils import trim_tfr
from plots import plot_evoked_tfr
from specparam_utils import params_to_spectra
from settings import *

# settings - analysis (match to other analyses)
LOG_POWER = True # whether to log-transform power
METHOD = 'mean' # method for computing band power

# settings - figure
plt.style.use(MPLSTYLE)
X_LIMITS = [-0.5, 1.0]
Y_LIMITS = [-2.2, 2.2] # for time-series plots


def main():

    # display progress
    t_start = get_start_time()

    # identify / create directories
    dir_output = f"{PROJECT_PATH}/figures/main_figures"
    if not os.path.exists(dir_output): 
        os.makedirs(f"{dir_output}")

    # create figure
    figsize = [WIDTH['2col'], WIDTH['2col']/2.5]
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # loop over materials and plot
    for material, ax in zip(MATERIALS, axes):
        plot_material(fig, ax, material)

    # add figure panel labels
    fig.text(0.05, 0.97, 'a', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.55, 0.97, 'b', fontsize=PANEL_FONTSIZE, fontweight='bold')

    # save figure
    fig.savefig(f"{dir_output}/figure_7-2", bbox_inches='tight')
    fig.savefig(f"{dir_output}/figure_7-2.png", bbox_inches='tight')

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


def plot_material(fig, ax, material):
    # load stats
    fname = f"{PROJECT_PATH}/data/results/band_power_statistics.csv"
    df_stats = pd.read_csv(fname, index_col=0)
    df_stats = df_stats.loc[(df_stats['memory']=='hit') & \
                            (df_stats['material']==material)]
    df_stats['sig_all'] = df_stats[[f'{band}_sig' for band in BANDS]].all(axis=1)
    df_stats = df_stats.loc[df_stats['sig_all']].reset_index(drop=True)

    # # load TFR and models and compute aperiodic-corrected TFR
    tfr_list = []
    for _, row in df_stats.iterrows():
        # load TFR
        fname = f"{row['patient']}_{material}_hit_chan{row['chan_idx']}_tfr.npz"
        data_in = np.load(f"{PROJECT_PATH}/data/ieeg_tfr/{fname}")
        tfr = np.nanmean(np.squeeze(data_in['tfr']), axis=0)
        tfr, freq, _ = trim_tfr(tfr, data_in['freq'], data_in['time'],
                                freq_range=FREQ_RANGE)
        # load model and extract aperiodic component
        fname = f"{row['patient']}_{material}_hit_chan{row['chan_idx']}_tfr_param_knee.json"
        model = SpectralGroupModel()
        model.load(f"{PROJECT_PATH}/data/ieeg_tfr_param/{fname}")
        tfr_ap = params_to_spectra(model, component='aperiodic').T

        # subtract aperiodic component and append to list        
        tfr_list.append(tfr - tfr_ap)
    
    # average over channels
    tfr = np.nanmean(np.array(tfr_list), axis=0) # average over channels and materials

    # plot
    tfr, _, time = trim_tfr(tfr, freq, data_in['time'], time_range=X_LIMITS)
    plot_evoked_tfr(tfr, freq, time, fig=fig, ax=ax, annotate_zero=True, 
                    cbar_label='power (au)')
    ax.set_title(f"{material.capitalize()[:-1]}-encoding")
    ax.set(xlabel='time (s)', ylabel='frequency (Hz)')


if __name__ == "__main__":
    main()
