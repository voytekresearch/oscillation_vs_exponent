"""
Plot histogram of intersection frequency. One subplot with all 
electrodes and another with task-modulated electrodes.
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
from settings import COLORS


def main():
    # display progress
    t_start = get_start_time()

    # identify / create directories
    dir_output = f"{PROJECT_PATH}/figures/intersection"
    if not os.path.exists(dir_output): 
        os.makedirs(f"{dir_output}")
            
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

    # create figure
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=[8, 4])

    # subplot 1 =====
    bin_edges = np.linspace(0,100,11)
    ax1.hist(f_rot_all, bins=bin_edges, alpha=0.6, color='grey', label='all electrodes')
    ax1.hist(f_rot_sig, bins=bin_edges, alpha=0.6, color='k', label='electrodes with\n reported effects')
    ax1.legend()

    # subplot 2 =====
    ax2.hist(intersection['words'][df['sig_all']], bins=bin_edges, alpha=0.8, label='word block', color=COLORS['brown'])
    ax2.hist(intersection['faces'][df['sig_all']], bins=bin_edges, alpha=0.8, label='face block', color=COLORS['blue'], zorder=0)
    ax2.axvline(np.nanmedian(f_rot_sig), linestyle='--', color='k',label='median', zorder=2)

    # add legend
    order = [0,1,2]
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend([handles[idx] for idx in order],[labels[idx] for idx in order])

    # axis labels
    for ax in[ax1, ax2]:
        ax.set_xlabel('intersection frequency (Hz)')
        ax.set_ylabel('electrode count')

    # add titles
    # fig.suptitle('Intersection of Power Spectra: Baseline v. Encoding')
    ax1.set_title('All Electrodes')
    ax2.set_title('Task-modulated electrodes')
        
    # save fig
    fig.savefig(f"{dir_output}/intersection_freq_histograms.png")

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


if __name__ == "__main__":
    main()
