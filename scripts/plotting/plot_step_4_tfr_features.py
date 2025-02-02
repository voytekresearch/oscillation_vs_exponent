"""

"""

# Imports - standard
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from specparam import SpectralGroupModel
from specparam.utils import trim_spectrum
from scipy import stats

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from info import MATERIALS
from utils import get_start_time, print_time_elapsed, confidence_interval
from tfr_utils import trim_tfr, subtract_baseline
from tfr_utils import zscore_tfr as zscore
from plots import plot_evoked_tfr
from settings import BANDS, AP_MODE, FREQ_RANGE, BCOLORS
from specparam_utils import compute_band_power
from specparam_utils import _compute_adjusted_band_power as compute_adjusted_band_power

# settings - example data to plot
PATIENT = 'pat11'
MATERIAL = 'words'
CHANNEL = 34

# settings - analysis
LOG_POWER = True # log-transform power
METHOD = 'mean' # method for computing band power

# settings - figure
plt.style.use('mplstyle/default.mplstyle')
FIGSIZE = (6.5, 4)
X_LIMITS = [-0.5, 1.0] # for time-series plots
Y_LIMITS = [[-4, 4], [-2.5, 2.5]] # [[electrode], [group]]


def main():

    # display progress
    t_start = get_start_time()

    # identify / create directories
    dir_output = f"{PROJECT_PATH}/figures/main_figures"
    if not os.path.exists(dir_output): 
        os.makedirs(f"{dir_output}")

    # create figure
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE, constrained_layout=True)

    # Plot single-electrode TFR ================================================
    fname = f'{PATIENT}_{MATERIAL}_hit_chan{CHANNEL}_tfr.npz'
    data_in = np.load(f"{PROJECT_PATH}/data/ieeg_tfr/{fname}")
    tfr_mean = np.nanmedian(np.squeeze(data_in['tfr']), axis=0) # average over trials
    tfr, freq, time = trim_tfr(tfr_mean, data_in['freq'], data_in['time'], 
                                freq_range=FREQ_RANGE, time_range=X_LIMITS)
    plot_evoked_tfr(tfr, freq, time, fig=fig, ax=axes[0,0], annotate_zero=True, 
                    cbar_label='power (au)', title='Single electrode')
    
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
        freq, spectra = trim_spectrum(data_in['freq'], tfr.T, FREQ_RANGE)
        power_adj[band] = compute_adjusted_band_power(freq, spectra, 
                                                      sm, f_range, 
                                                      method=METHOD, 
                                                      log_power=LOG_POWER)

    # plot power (after z-scoring and subtracting baseline)
    for pow, ax in zip([power, power_adj], [axes[0,1], axes[0,2]]):
        for band in BANDS.keys():
            ts = stats.zscore(pow[band], nan_policy='omit')
            ts = np.squeeze(subtract_baseline(ts[np.newaxis, :], time, 
                                              t_baseline=[X_LIMITS[0], 0]))
            ax.plot(time, ts, label=band, color=BCOLORS[band])

    # plot exponent (after z-scoreing and subtracting baseline)
    exponent = stats.zscore(exponent, nan_policy='omit')
    exponent = np.squeeze(subtract_baseline(exponent[np.newaxis,:], time, 
                                            t_baseline=[X_LIMITS[0], 0]))
    axes[0,2].plot(time, exponent, label='exponent', color=BCOLORS['exponent'])

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
                
                freq, spectra = trim_spectrum(data_in['freq'], tfr.T, FREQ_RANGE)
                temp = compute_adjusted_band_power(freq, spectra, sm, 
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
            ci = confidence_interval(pow[band])
            ax.plot(time, np.nanmean(pow[band], axis=0), label=band, color=BCOLORS[band])
            ax.fill_between(time, ci[0], ci[1], alpha=0.2, color=BCOLORS[band])

    # plot exponent (after z-scoreing and subtracting baseline)
    exponent = subtract_baseline(zscore(np.array(exp_list)), time, 
                                 t_baseline=[X_LIMITS[0], 0])
    ci = confidence_interval(exponent)
    axes[1,2].plot(time, np.nanmean(exponent, axis=0), label='exponent', color=BCOLORS['exponent'])
    axes[1,2].fill_between(time, ci[0], ci[1], alpha=0.2, color=BCOLORS['exponent'])

    # label and adjust plots ===================================================
    for row in [0, 1]:
        # set title
        axes[row, 1].set_ylabel('total power (au)')
        axes[row, 2].set_ylabel('adjusted power (au)')

        for ax in axes[row, 1:3]:
            # set y labels and limits
            ax.set_ylim(Y_LIMITS[row])
            
            # set x labels and limits
            ax.axhline(0, color='k', linestyle='--')
            ax.axvline(0, color='k', linestyle='--')
            ax.set_xlim(X_LIMITS)

            # add legend
            ax.legend(loc='upper left')

    # save figure
    fig.savefig(f"{dir_output}/tfr_features.png")

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


if __name__ == "__main__":
    main()
