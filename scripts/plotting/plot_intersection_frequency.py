"""
Plot intersection frequency results. Subplot A: grand average power spectra,
with the 95% confidence interval of the intersection frequency annotated.
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


def main():
    # display progress
    t_start = get_start_time()

    # identify / create directories
    dir_output = f"{PROJECT_PATH}/figures/intersection"
    if not os.path.exists(dir_output): 
        os.makedirs(f"{dir_output}")

    # Load data ================================================================
            
    # load rotation analysis results
    intersection = dict()
    for material in MATERIALS:
        fname = f"intersection_results_{material}_hit_knee.npz"
        data_in = np.load(f"{PROJECT_PATH}/data/ieeg_intersection_results/{fname}")
        intersection[material] = data_in['intersection']

    # load task-modulation results
    df = pd.read_csv(f"{PROJECT_PATH}/data/results/ieeg_modulated_channels.csv")

    # combine data for materials
    f_rot_all = np.concatenate([intersection['words'], intersection['faces']])
    f_rot_sig = np.concatenate([intersection['words'][df['sig_all']], 
                                intersection['faces'][df['sig_all']]])
    
    # compute confidence interval for intersection frequency
    scale = np.nanstd(f_rot_sig)/np.sqrt(len(f_rot_sig))
    ci = norm.interval(0.95, loc=np.nanmean(f_rot_sig), scale=scale)

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

    # subplot 1: grand average power spectra ===================================

    # create figure
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=[8, 4])

    # plot spectra
    ax1.set_title('Grand average PSD')
    plot_spectra_2conditions(spectra_pre, spectra_post, freq, ax=ax1,
                                color=[COLORS['light_brown'], COLORS['brown']])
    ax1.set_xlim([4, 100]) # SpecParam fitting range

    # annotate 95% confidence interval
    ax1.axvspan(ci[0], ci[1], color='grey', alpha=0.3)

    # subplot 2: histogram of intersection frequency ===========================
    
    # plot histogram
    bin_edges = np.linspace(0, 100, 11)
    ax2.hist(intersection['words'][df['sig_all']], bins=bin_edges, alpha=0.8, 
             label='word block', color=COLORS['brown'])
    ax2.hist(intersection['faces'][df['sig_all']], bins=bin_edges, 
             alpha=0.8, label='face block', color=COLORS['blue'], zorder=0)
    ax2.axvline(np.nanmedian(f_rot_sig), linestyle='--', color='k', 
                label='median', zorder=2)

    # label plot
    ax2.set_title('Intersection frequency')
    ax2.set_xlabel('intersection frequency (Hz)')
    ax2.set_ylabel('electrode count')
    ax2.legend()

    # save fig
    fig.savefig(f"{dir_output}/intersection_freq_histograms.png")

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


if __name__ == "__main__":
    main()
