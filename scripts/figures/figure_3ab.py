"""
Plot example data. Plot raw time-series and power spectra for an example channel.
Subplot 1: Raw time-series for 5 examples trials
Subplot 2: Power spectra for baseline and encoding periods with intersection 
frequency highlighted.
"""

# Imports - standard
import os
import numpy as np
import pandas as pd
from mne import read_epochs
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from utils import get_start_time, print_time_elapsed
from settings import FREQ_RANGE, WIDTH, BANDS, BCOLORS, COLORS, PANEL_FONTSIZE
from plots import beautify_ax

# settings
plt.style.use('mplstyle/nature_neuro.mplstyle')

# settings - example data to visualize
PATIENT = 'pat11'
CHAN_IDX = 35
MATERIAL = 'faces'

def main():

    # display progress
    t_start = get_start_time()

    # identify / create directories
    dir_output = f"{PROJECT_PATH}/figures/main_figures"
    if not os.path.exists(dir_output): 
        os.makedirs(f"{dir_output}")

    # load data ================================================================

    # load iEEG time-series results
    fname_in = f"{PATIENT}_{MATERIAL}_hit_epo.fif"
    epochs = read_epochs(f"{PROJECT_PATH}/data/ieeg_epochs/{fname_in}")
    signals = epochs.get_data()
    signal = signals[:, CHAN_IDX]
    time = epochs.times

    # plot data =================================================================

    # create gridspec and nested gridspec for subplots
    fig = plt.figure(figsize=[WIDTH['2col'], WIDTH['2col']/4])
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2.75, 2])
    gs2a = gridspec.GridSpecFromSubplotSpec(5,1, subplot_spec=gs[0])
    gs2bc = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec=gs[1])

    # ==================== Fig 2a ====================
    # plot raw time-series (baseline and encoding)

    # sererate baseline and encoding
    mask_b = (time>=-1)&(time<=0)
    mask_e = (time>=0)&(time<=1)

    # create subplots
    ax0 = fig.add_subplot(gs2a[0,:])
    ax1 = fig.add_subplot(gs2a[1,:])
    ax2 = fig.add_subplot(gs2a[2,:])
    ax3 = fig.add_subplot(gs2a[3,:])
    ax4 = fig.add_subplot(gs2a[4,:])
    axes = [ax0,ax1,ax2,ax3,ax4]

    # plot 5 trials
    trials = np.random.randint(0, len(signal), 5)
    for trial, ax in zip(trials, axes):
        # plot
        ax.plot(time[mask_b], signal[trial, mask_b], 
                color='k', linewidth=1)
        ax.plot(time[mask_e], signal[trial, mask_e], 'k', linewidth=1)

    # add ylabel across these 5 subplots, 'trials'. rotate it.
    fig.text(0.0, 0.5, 'trials', va='center', rotation='vertical', fontsize=6)

    # remove cluttered axes, ticks, and spines
    ax4.axes.yaxis.set_ticks([])
    for ax in axes[:4]:
        ax.axis('off')
    for loc in ['top','right','left']:
        ax4.spines[loc].set_visible(False)

    # label
    ax0.set_title('Raw iEEG time-series')
    ax4.set_xlabel('time relative to stimulus onset (s)')

    # ==================== Fig 2b and 2d ====================
    ax2b = fig.add_subplot(gs2bc[0])
    ax2c = fig.add_subplot(gs2bc[1])

    for ax, material, color in zip([ax2b, ax2c], 
                                   ['words', 'faces'], 
                                   ['brown', 'blue']):
        plot_spectra(ax, material, color)
    
    # add joint title centered over PSD subplots (B)
    fig.text(0.8, 0.97, 'Power spectra', ha='center', 
             va='center', fontsize=7)
    
    # add subplot header
    line = plt.Line2D((0.1, 0.9), (1.01, 1.01), color='black', linewidth=1, 
                    transform=fig.transFigure, figure=fig)
    fig.add_artist(line)
    fig.text(0.5, 1.07, "Single-electrode example", ha='center', va='top', 
             fontsize=7, fontdict={'fontweight': 'bold'})

    # add figure panel labels
    fig.text(0.01, 0.92, 'a', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.60, 0.92, 'b', fontsize=PANEL_FONTSIZE, fontweight='bold')

    # save figure
    fig.savefig(f"{dir_output}/figure_3ab.png", bbox_inches='tight')

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


def plot_spectra(axi, material, color):
    # load spectral results
    fname_in = '%s_%s_hit_XXXstim_psd.npz' %(PATIENT, material)
    psd_pre_in = np.load(f"{PROJECT_PATH}/data/ieeg_psd/{fname_in.replace('XXX','pre')}")
    psd_post_in = np.load(f"{PROJECT_PATH}/data/ieeg_psd/{fname_in.replace('XXX','post')}")
    psd_pre_all = psd_pre_in['psd'][:,CHAN_IDX]
    psd_post_all = psd_post_in['psd'][:,CHAN_IDX]
    freq = psd_pre_in['freq']

    # calc confidence interval for spectra (across trials)
    conf_pre = stats.norm.interval(0.95, loc=np.mean(psd_pre_all, 0),
        scale=np.std(psd_pre_all, 0)/np.sqrt(len(psd_pre_all)))
    conf_post = stats.norm.interval(0.95, loc=np.mean(psd_post_all, 0),
        scale=np.std(psd_post_all, 0)/np.sqrt(len(psd_post_all)))

    # Plot spectra
    axi.loglog(freq, np.nanmean(psd_pre_all, 0), label='baseline',
                color=COLORS[f'light_{color}'], linewidth=1)
    axi.loglog(freq, np.nanmean(psd_post_all, 0), label='encoding',
                color=COLORS[color], linewidth=1)
    axi.set_xlim(FREQ_RANGE)

    # plot 95% confidence intrval
    axi.fill_between(freq, conf_pre[0], conf_pre[1], edgecolor=None,
                    color=COLORS[f'light_{color}'], alpha=0.5)
    axi.fill_between(freq, conf_post[0], conf_post[1], edgecolor=None,
                        color=COLORS[color], alpha=0.5)

    # shade oscillation bands
    for band in ['alpha', 'gamma']:
        axi.axvspan(BANDS[band][0], BANDS[band][1], facecolor=BCOLORS[band],
                    alpha=0.4)

    # subplot 2 - label
    axi.set_title(f'\n{material[:-1]}-encoding')
    axi.set_xlabel('frequency (Hz)')
    axi.set_ylabel('power ($\u03bcV^2/Hz$)')
    axi.legend(loc='lower left')
    axi.axvline(1, color='gray', linestyle='--', linewidth=1)

    # beautify
    beautify_ax(axi)


if __name__ == "__main__":
    main()
