"""

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
from utils import get_start_time, print_time_elapsed
from tfr_utils import trim_tfr
from plots import plot_evoked_tfr
from settings import BANDS

# settings - example data to plot
PATIENT = 'pat11'
MATERIAL = 'words'
CHANNEL = 34

def main():

    # display progress
    t_start = get_start_time()

    # identify / create directories
    dir_output = f"{PROJECT_PATH}/figures/main_figures"
    if not os.path.exists(dir_output): 
        os.makedirs(f"{dir_output}")

    # create figure
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Plot single-electrode TFR ================================================
    fname = f'{PATIENT}_{MATERIAL}_hit_chan{CHANNEL}_tfr.npz'
    data_in = np.load(f"{PROJECT_PATH}/data/ieeg_tfr/{fname}")
    tfr_mean = np.nanmedian(np.squeeze(data_in['tfr']), axis=0) # average over trials
    tfr, freq, time = trim_tfr(tfr_mean, data_in['freq'], data_in['time'], 
                                freq_range=[0, 100], time_range=[-0.5, 1.0])
    plot_evoked_tfr(tfr, freq, time, fig=fig, ax=axes[0,0], annotate_zero=True, 
                    cbar_label='power (z-score)', title='Single electrode')
    
    # Plot single-electrode time-series ========================================


    # Plot group TFR ===========================================================
    # load stats
    fname = f"{PROJECT_PATH}/data/results/ieeg_modulated_channels.csv"
    stats = pd.read_csv(fname, index_col=0)
    stats = stats.loc[stats['sig_all']].reset_index(drop=True)

    # load TFR for active channels 
    tfr_list = []
    for _, row in stats.iterrows():
        for material in MATERIALS:
            fname = f"{row['patient']}_{material}_hit_chan{row['chan_idx']}_tfr.npz"
            data_in = np.load(f"{PROJECT_PATH}/data/ieeg_tfr/{fname}")
            tfr_list.append(np.nanmedian(np.squeeze(data_in['tfr']), axis=0))
    tfr = np.nanmean(np.array(tfr_list), axis=0) # average over channels and materials

    # plot
    tfr, freq, time = trim_tfr(tfr, data_in['freq'], data_in['time'], 
                                freq_range=[0, 100], time_range=[-0.5, 1.0])
    plot_evoked_tfr(tfr, freq, time, fig=fig, ax=axes[1,0], annotate_zero=True, 
                    cbar_label='power (z-score)', title='Group average')

    # Plot group time-series ===================================================
    

    # save figure
    fig.savefig(f"{dir_output}/tfr_features.png", dpi=300)

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


if __name__ == "__main__":
    main()
