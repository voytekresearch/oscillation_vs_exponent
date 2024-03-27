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
from scipy.stats import norm

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from utils import get_start_time, print_time_elapsed
from info import MATERIALS
from settings import COLORS
from plots import plot_spectra_2conditions

# settings
FIGSIZE = [5, 2]


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
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=FIGSIZE, 
                                   constrained_layout=True)

    # plot spectra
    plot_spectra_2conditions(spectra_pre, spectra_post, freq, ax=ax1,
                                color=[COLORS['light_brown'], COLORS['brown']])
    ax1.set_title('Grand average')
    ax1.set_xlim([4, 100]) # SpecParam fitting range

    # plot histogram
    ax2.set_title('Baseline v. encoding')
    bin_edges = np.linspace(0, 100, 11)
    ax2.hist(f_intersection, bins=bin_edges, color=COLORS['brown'])
    ax2.set_xlabel('intersection frequency (Hz)')
    ax2.set_ylabel('electrode count')
    ax2.axvline(np.nanmedian(f_intersection), color='k', linestyle='--')
    ax2.text(0.7, 0.9, f"median={int(np.nanmedian(f_intersection))} Hz", 
            transform=ax2.transAxes, ha='center', va='center')

    # save fig
    fig.savefig(f"{dir_output}/intersection_frequency.png")

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


if __name__ == "__main__":
    main()
