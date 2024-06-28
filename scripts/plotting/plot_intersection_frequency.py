"""
Plot intersection frequency results. 

Subplot A: grand average power spectra.
Subplot B: histogram of intersection frequency, for each condition.
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
from utils import get_start_time, print_time_elapsed
from info import MATERIALS
from settings import FREQ_RANGE, RGB, WIDTH
from plots import plot_spectra_2conditions, beautify_ax

# settings
plt.style.use('mplstyle/default.mplstyle')

def main():
    # display progress
    t_start = get_start_time()

    # identify / create directories
    dir_output = f"{PROJECT_PATH}/figures/main_figures"
    if not os.path.exists(dir_output): 
        os.makedirs(f"{dir_output}")

    # load task-modulation results
    df = pd.read_csv(f"{PROJECT_PATH}/data/results/ieeg_modulated_channels.csv")

    # load rotation analysis results
    intersection = dict()
    for material in MATERIALS:
        fname = f"intersection_results_{material}_hit_knee.npz"
        data_in = np.load(f"{PROJECT_PATH}/data/ieeg_intersection_results/{fname}")
        intersection[material] = data_in['intersection']
    f_intersection = np.concatenate([intersection['words'][df['sig_all']], 
                                     intersection['faces'][df['sig_all']]])
    
    # load spectal results
    psd = dict()
    for material in MATERIALS:
        for epoch in ['pre','post']:
            fname = f"psd_{material}_hit_{epoch}stim.npz"
            data_in = np.load(f"{PROJECT_PATH}/data/ieeg_spectral_results/{fname}")
            psd[(material, epoch)] = data_in['spectra'][df['sig_all']]
    freq = data_in['freq']

    # compute grand average spectra
    spectra_pre = np.concatenate([psd[material, 'pre'] for material in MATERIALS])
    spectra_post = np.concatenate([psd[material, 'post'] for material in MATERIALS])

    # create figure
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=[WIDTH['1col'], WIDTH['1col']/2], 
                                   constrained_layout=True)

    # plot spectra
    plot_spectra_2conditions(spectra_pre, spectra_post, freq, ax=ax1,
                                color=['grey', 'k'],
                                shade_sem=False)
    ax1.set_title('Grand average')
    ax1.set_xlim(FREQ_RANGE)
    ax1.grid(False)

    # annotate intersection frequency
    median_f = np.nanmedian(f_intersection)
    median_idx = np.argmin(np.abs(freq - median_f))
    ax1.scatter(freq[median_idx], np.nanmean(spectra_pre[:, median_idx]),
                color=RGB[2], s=20, zorder=10)
    ax1.legend(['baseline', 'encoding', 'intersection'], loc='lower left')

    # plot histogram
    ax2.set_title('Intersection frequency')
    bins = np.linspace(0, 100, 11)
    ax2.hist(f_intersection, bins, color=RGB[2])
    ax2.set_xlabel('frequency (Hz)')
    ax2.set_ylabel('electrode count')

    # remove top and right spines
    for ax in [ax1, ax2]:
        beautify_ax(ax)

    # save fig
    fig.savefig(f"{dir_output}/intersection_frequency")
    fig.savefig(f"{dir_output}/intersection_frequency.png")

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


if __name__ == "__main__":
    main()
