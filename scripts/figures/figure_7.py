"""

"""

# Imports - standard
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from specparam import SpectralGroupModel
from specparam.utils import trim_spectrum
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from info import MATERIALS
from utils import get_start_time, print_time_elapsed, confidence_interval
from tfr_utils import trim_tfr, subtract_baseline
from tfr_utils import zscore_tfr as zscore
from plots import plot_evoked_tfr, beautify_ax
from settings import BANDS, AP_MODE, FREQ_RANGE, BCOLORS, WIDTH, PANEL_FONTSIZE
from specparam_utils import compute_band_power
from specparam_utils import _compute_adjusted_band_power as compute_adjusted_band_power

# settings - analysis (match to other analyses)
LOG_POWER = True # whether to log-transform power
METHOD = 'mean' # method for computing band power

# settings - figure
plt.style.use('mplstyle/nature_neuro.mplstyle')
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
    figsize = [WIDTH['2col'], WIDTH['2col']/1.5]
    fig = plt.figure(figsize=figsize, tight_layout=True)
    spec = gridspec.GridSpec(figure=fig, ncols=3, nrows=3,
                            height_ratios=[1, 0.1, 1])
    
    # loop over materials and plot
    for material, row in zip(MATERIALS, [0, 2]):
        axes = [fig.add_subplot(spec[row, 0]),
                fig.add_subplot(spec[row, 1]),
                fig.add_subplot(spec[row, 2])]
        plot_material(fig, axes, material)

    # add section titles and line between subplot rows
    ax_header = fig.add_subplot(spec[2, :])
    ax_header.axis('off')
    for ypos in [0.48, 1.0]:
        line = plt.Line2D((0.1, 0.9), (ypos, ypos), color='black', linewidth=1, 
                        transform=fig.transFigure, figure=fig)
        fig.add_artist(line)
    fig.text(0.5, 1.03, "Word-encoding", ha='center', va='top', fontsize=7, 
             fontdict={'fontweight': 'bold'})
    fig.text(0.5, 0.51, "Face-encoding", ha='center', va='top', fontsize=7, 
             fontdict={'fontweight': 'bold'})

    # add figure panel labels
    fig.text(0.05, 0.97, 'a.', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.38, 0.97, 'b.', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.71, 0.97, 'c.', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.05, 0.44, 'd.', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.38, 0.44, 'e.', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.71, 0.44, 'f.', fontsize=PANEL_FONTSIZE, fontweight='bold')

    # save figure
    fig.savefig(f"{dir_output}/figure_7", bbox_inches='tight')
    fig.savefig(f"{dir_output}/figure_7.png", bbox_inches='tight')

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


def plot_material(fig, axes, material):
    # Plot normalized spectrogram ==============================================

    # load stats
    fname = f"{PROJECT_PATH}/data/results/band_power_statistics.csv"
    df_stats = pd.read_csv(fname, index_col=0)
    df_stats = df_stats.loc[(df_stats['memory']=='hit') & \
                            (df_stats['material']==material)]
    df_stats['sig_all'] = df_stats[[f'{band}_sig' for band in BANDS]].all(axis=1)
    df_stats = df_stats.loc[df_stats['sig_all']].reset_index(drop=True)

    # # load TFR for active channels 
    tfr_list = []
    for _, row in df_stats.iterrows():
        fname = f"{row['patient']}_{material}_hit_chan{row['chan_idx']}_tfr.npz"
        data_in = np.load(f"{PROJECT_PATH}/data/ieeg_tfr/{fname}")
        tfr_list.append(np.nanmedian(np.squeeze(data_in['tfr']), axis=0))
    tfr = np.nanmean(np.array(tfr_list), axis=0) # average over channels and materials

    # plot
    tfr, freq, time = trim_tfr(tfr, data_in['freq'], data_in['time'], 
                                freq_range=FREQ_RANGE, time_range=X_LIMITS)
    plot_evoked_tfr(tfr, freq, time, fig=fig, ax=axes[0], annotate_zero=True, 
                    cbar_label='power (au)')

    # Plot time-series ========================================================

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
    for pow, ax, tag in zip([power, power_adj], [axes[1], axes[2]], 
                            ['total', 'adjusted']):
        for band in BANDS.keys():
            ci = confidence_interval(pow[band])
            ax.plot(time, np.nanmean(pow[band], axis=0), 
                    label=f"{tag} {band}", color=BCOLORS[band])
            ax.fill_between(time, ci[0], ci[1], alpha=0.2, color=BCOLORS[band])

    # plot exponent (after z-scoreing and subtracting baseline)
    exponent = subtract_baseline(zscore(np.array(exp_list)), time, 
                                 t_baseline=[X_LIMITS[0], 0])
    ci = confidence_interval(exponent)
    axes[2].plot(time, np.nanmean(exponent, axis=0), label='exponent', 
                 color=BCOLORS['exponent'])
    axes[2].fill_between(time, ci[0], ci[1], alpha=0.2, 
                         color=BCOLORS['exponent'])

    # add statistical annotations ==============================================
    time_ = time[(time > 0) & (time < X_LIMITS[1])] 
    
    for ii, (feature, values) in enumerate(zip(['alpha', 'gamma'],
                                             [power['alpha'], power['gamma']])):
        values = values[:, ((time > 0) & (time < X_LIMITS[1]))]
        sig = compute_significance(values)
        for jj in range(len(sig)-1):
            if sig[jj] and sig[jj+1]:
                axes[1].plot([time_[jj], time_[jj+1]], 
                             [Y_LIMITS[1]-0.2-(0.1*ii)]*2, 
                             color=BCOLORS[feature], linewidth=2)

    for ii, (feature, values) in enumerate(zip(['exponent', 'alpha', 'gamma'],
                                             [exponent, power_adj['alpha'], 
                                              power_adj['gamma']])):
        values = values[:, ((time > 0) & (time < X_LIMITS[1]))]
        sig = compute_significance(values)
        for jj in range(len(sig)-1):
            if sig[jj] and sig[jj+1]:
                axes[2].plot([time_[jj], time_[jj+1]], 
                             [Y_LIMITS[1]-0.2-(0.1*ii)]*2, 
                             color=BCOLORS[feature], linewidth=2)

    # label and adjust plots ===================================================

    # set title
    axes[0].set_title('Normalized spectrogram')
    axes[1].set_title('Total power')
    axes[2].set_title('Spectral parameters')

    # set x labels
    for ax in axes:
        ax.set_xlabel('time (s)')

    # adjust time-series plots
    for ax in axes[1:3]:
        # label axes
        ax.set_ylabel('z-score')

        # annotate zero
        ax.axhline(0, color='k', linestyle='--')
        ax.axvline(0, color='k', linestyle='--')

        # set axis limits
        ax.set_xlim(X_LIMITS)
        ax.set_ylim(Y_LIMITS)

        # add legend
        ax.legend(facecolor='white', framealpha=1,
                  bbox_to_anchor=(0.001, 0.001), loc='lower left')

        # remove right and top spines
        beautify_ax(ax)


def compute_significance(values):
    pvalues = []
    for ii in range(values.shape[1]):
        res_i = stats.ttest_1samp(values[:, ii], 0, nan_policy='omit')
        pvalues.append(res_i.pvalue)
    pvalues = np.array(pvalues)
    mt = multipletests(pvalues, alpha=0.05, method='holm')
    sig = mt[0]

    return sig


if __name__ == "__main__":
    main()
