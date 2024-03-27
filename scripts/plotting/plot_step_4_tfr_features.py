"""

"""

# Imports - standard
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from specparam import SpectralGroupModel
from scipy import stats

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from info import MATERIALS
from utils import get_start_time, print_time_elapsed
from tfr_utils import trim_tfr, subtract_baseline
from tfr_utils import zscore_tfr as zscore
from plots import plot_evoked_tfr
from settings import BANDS, AP_MODE, FREQ_RANGE
from specparam_utils import compute_band_power, compute_adjusted_band_power

# settings - example data to plot
PATIENT = 'pat11'
MATERIAL = 'words'
CHANNEL = 34

# settings - analysis
LOG_POWER = True # log-transform power
METHOD = 'sum' # method for computing band power

# settings - figure
FIGSIZE = (10, 4)
X_LIMITS = [-0.25, 1.0] # for time-series plots
Y_LIMITS = [[-3, 3], [-2, 2]] # [[electrode], [group]]


def main():

    # display progress
    t_start = get_start_time()

    # identify / create directories
    dir_output = f"{PROJECT_PATH}/figures/main_figures"
    if not os.path.exists(dir_output): 
        os.makedirs(f"{dir_output}")

    # create figure
    fig, axes = plt.subplots(2, 4, figsize=FIGSIZE, constrained_layout=True)

    # Plot single-electrode TFR ================================================
    fname = f'{PATIENT}_{MATERIAL}_hit_chan{CHANNEL}_tfr.npz'
    data_in = np.load(f"{PROJECT_PATH}/data/ieeg_tfr/{fname}")
    tfr_mean = np.nanmedian(np.squeeze(data_in['tfr']), axis=0) # average over trials
    tfr, freq, time = trim_tfr(tfr_mean, data_in['freq'], data_in['time'], 
                                freq_range=FREQ_RANGE, time_range=X_LIMITS)
    plot_evoked_tfr(tfr, freq, time, fig=fig, ax=axes[0,0], annotate_zero=True, 
                    cbar_label='power (z-score)', title='Single electrode')
    
    # Plot single-electrode time-series ========================================
    # load SpecParam results
    fname = f'{PATIENT}_{MATERIAL}_hit_chan{CHANNEL}_tfr_param_knee'
    sm = SpectralGroupModel()
    sm.load(f"{PROJECT_PATH}/data/ieeg_tfr_param/{fname}")
    exponent = sm.get_params('aperiodic','exponent')

    # load spectral results and compute band power
    fname = f'{PATIENT}_{MATERIAL}_hit_chan{CHANNEL}_tfr.npz'
    data_in = np.load(f"{PROJECT_PATH}/data/ieeg_tfr/{fname}")
    tfr = np.nanmean(data_in['tfr'], axis=0)
    time = data_in['time']
    power = dict()
    power_adj = dict()
    for band, f_range in BANDS.items():
        power[band] = compute_band_power(data_in['freq'], tfr.T, f_range, 
                                         method=METHOD, log_power=LOG_POWER)
        power_adj[band] = compute_adjusted_band_power(data_in['freq'], tfr.T, 
                                                      sm, f_range, 
                                                      method=METHOD, 
                                                      log_power=LOG_POWER)

    # plot power (after z-scoring and subtracting baseline)
    for pow, ax in zip([power, power_adj], [axes[0,1], axes[0,2]]):
        for band in BANDS.keys():
            ts = stats.zscore(pow[band], nan_policy='omit')
            ts = np.squeeze(subtract_baseline(ts[np.newaxis, :], time, 
                                              t_baseline=[X_LIMITS[0], 0]))
            ax.plot(time, ts, label=band)

    # plot exponent (after z-scoreing and subtracting baseline)
    exponent = stats.zscore(exponent, nan_policy='omit')
    exponent = np.squeeze(subtract_baseline(exponent[np.newaxis,:], time, 
                                            t_baseline=[X_LIMITS[0], 0]))
    axes[0,2].plot(time, exponent, label='exponent')

    # # Plot group TFR ===========================================================
    # load stats
    fname = f"{PROJECT_PATH}/data/results/ieeg_modulated_channels.csv"
    df_stats = pd.read_csv(fname, index_col=0)
    df_stats = df_stats.loc[df_stats['sig_all']].reset_index(drop=True)

    # load TFR for active channels 
    tfr_list = []
    for _, row in df_stats.iterrows():
        for material in MATERIALS:
            fname = f"{row['patient']}_{material}_hit_chan{row['chan_idx']}_tfr.npz"
            data_in = np.load(f"{PROJECT_PATH}/data/ieeg_tfr/{fname}")
            tfr_list.append(np.nanmedian(np.squeeze(data_in['tfr']), axis=0))
    tfr = np.nanmean(np.array(tfr_list), axis=0) # average over channels and materials

    # plot
    tfr, freq, time = trim_tfr(tfr, data_in['freq'], data_in['time'], 
                                freq_range=FREQ_RANGE, time_range=X_LIMITS)
    plot_evoked_tfr(tfr, freq, time, fig=fig, ax=axes[1,0], annotate_zero=True, 
                    cbar_label='power (z-score)', title='Group average')

    # Plot group time-series ===================================================
    # initialize lists
    exp_list = []
    power = dict()
    power_adj = dict()
    for band in BANDS.keys():
        power[band] = []
        power_adj[band] = []

    # aggregate data for all active channels
    for _, row in df_stats.iterrows():
        for material in MATERIALS:
            # load exponent
            fname = f"{row['patient']}_{material}_hit_chan{row['chan_idx']}_tfr_param_{AP_MODE}"
            sm = SpectralGroupModel()
            sm.load(f"{PROJECT_PATH}/data/ieeg_tfr_param/{fname}")
            exp_list.append(sm.get_params('aperiodic','exponent'))
            
            # load tfr and compute band power
            fname = f"{row['patient']}_{material}_hit_chan{row['chan_idx']}_tfr.npz"
            data_in = np.load(f"{PROJECT_PATH}/data/ieeg_tfr/{fname}")
            tfr = np.nanmedian(np.squeeze(data_in['tfr']), axis=0)
            
            for band, f_range in BANDS.items():
                temp = compute_band_power(data_in['freq'], tfr.T, f_range, 
                                          method=METHOD, log_power=LOG_POWER)
                power[band].append(temp)
                temp = compute_adjusted_band_power(data_in['freq'], tfr.T, sm, 
                                                   f_range, method=METHOD, 
                                                   log_power=LOG_POWER)
                power_adj[band].append(temp)
    
    # z-score power and subtract baseline
    time = data_in['time']
    for band in BANDS.keys():
        # convert to arrays
        power[band] = np.array(power[band])
        power_adj[band] = np.array(power_adj[band])

        # z-score
        power[band] = zscore(power[band])
        power_adj[band] = zscore(power_adj[band])

        # subtract baseline
        power[band] = subtract_baseline(power[band], time, 
                                        t_baseline=[X_LIMITS[0], 0])
        power_adj[band] = subtract_baseline(power_adj[band], time, 
                                            t_baseline=[X_LIMITS[0], 0])

    # plot power
    for pow, ax in zip([power, power_adj], [axes[1,1], axes[1,2]]):
        for band in BANDS.keys():
            ci = compute_ci(pow[band])
            ax.plot(time, np.nanmean(pow[band], axis=0), label=band)
            ax.fill_between(time, ci[0], ci[1], alpha=0.2)

    # plot exponent (after z-scoreing and subtracting baseline)
    exponent = subtract_baseline(zscore(np.array(exp_list)), time, 
                                 t_baseline=[X_LIMITS[0], 0])
    ci = compute_ci(exponent)
    axes[1,2].plot(time, np.nanmean(exponent, axis=0), label='exponent')
    axes[1,2].fill_between(time, ci[0], ci[1], alpha=0.2)

    # label and adjust plots ===================================================
    for row in [0, 1]:
        # set title
        axes[row, 1].set_title('Total power')
        axes[row, 2].set_title('Adjusted power')

        # set y labels
        for ax in axes[row, 1:3]:
            ax.set_ylabel('power (z-score)')
            
        # set x labels and limits
        for ax in axes[row, 1:3]:
            ax.axhline(0, color='k', linestyle='--')
            ax.axvline(0, color='k', linestyle='--')
            ax.set_xlim(X_LIMITS)

        # adjust y limits
        for ax in axes[row, 1:3]:
            ax.set_ylim(Y_LIMITS[row])

    # add legend
    handles, labels = axes[1,2].get_legend_handles_labels()
    axes[0,3].axis('off')
    axes[1,3].axis('off')
    axes[0,3].legend(handles, labels, loc='center')

    # save figure
    fig.savefig(f"{dir_output}/tfr_features.png", dpi=300)

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


def compute_ci(data):
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    ci = stats.norm.interval(0.95, loc=mean, scale=std/np.sqrt(data.shape[0]))

    return ci


if __name__ == "__main__":
    main()
