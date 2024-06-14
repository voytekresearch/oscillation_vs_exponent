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
from settings import AP_MODE, COLORS, FREQ_RANGE
from info import MATERIALS

# settings
plt.style.use('mplstyle/default.mplstyle')
FIGSIZE = [6.5, 2]
color = 'blue'

# settings - example data to visualize
PATIENT = 'pat11'
CHAN_IDX = 34
MATERIAL = 'words'

def main():

    # display progress
    t_start = get_start_time()

    # identify / create directories
    dir_output = f"{PROJECT_PATH}/figures/main_figures"
    if not os.path.exists(dir_output): 
        os.makedirs(f"{dir_output}")

    # load data ================================================================

    # load channel info
    chan_info = pd.read_csv(f"{PROJECT_PATH}/data/ieeg_metadata/ieeg_channel_info.csv")

    # load rotation analysis results
    intersection = dict()
    intersection_idx = dict()
    for material in MATERIALS:
        fname = f"intersection_results_{material}_hit_{AP_MODE}.npz"
        data_in = np.load(f"{PROJECT_PATH}/data/ieeg_intersection_results/{fname}", allow_pickle=True)
        intersection[material] = data_in['intersection']
        intersection_idx[material] = data_in['intersection_idx']

    # load iEEG time-series results
    fname_in = f"{PATIENT}_{MATERIAL}_hit_epo.fif"
    epochs = read_epochs(f"{PROJECT_PATH}/data/ieeg_epochs/{fname_in}")
    signals = epochs.get_data()
    signal = signals[:, CHAN_IDX]
    time = epochs.times

    # load spectral results
    fname_in = '%s_%s_hit_XXXstim_psd.npz' %(PATIENT, MATERIAL)
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

    # get rotation frequency and index
    mask = (chan_info['patient']==PATIENT) & (chan_info['chan_idx']==CHAN_IDX)
    f_rotation = intersection['words'][mask][0]
    idx_rotation = np.argmin(np.abs(freq - f_rotation))

    # plot data =================================================================

    # create gridspec and nested gridspec for subplots
    fig = plt.figure(figsize=FIGSIZE)
    gs = gridspec.GridSpec(1,2, figure=fig, width_ratios=[2,1])
    gs2a = gridspec.GridSpecFromSubplotSpec(5,1, subplot_spec=gs[0], hspace=0)
    gs2b = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec=gs[1])

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
                color=COLORS[f'light_{color}'], linewidth=1)
        ax.plot(time[mask_e], signal[trial, mask_e], color=COLORS[color], 
                linewidth=1)

    # remove cluttered axes, ticks, and spines
    ax4.axes.yaxis.set_ticks([])
    for ax in axes[:4]:
        ax.axis('off')
    for loc in ['top','right','left']:
        ax4.spines[loc].set_visible(False)

    # label
    ax0.set_title('iEEG Time-series')
    ax4.set_xlabel('time relative to stimulus onset (s)')

    # ==================== Fig 2b ====================
    # create subplot
    ax2b = fig.add_subplot(gs2b[0,0])

    # Plot spectra
    ax2b.loglog(freq, np.nanmean(psd_pre_all, 0), color=COLORS[f'light_{color}'], 
                label='Baseline')
    ax2b.loglog(freq, np.nanmean(psd_post_all, 0), color=COLORS[color], 
                label='Encoding')
    ax2b.set_xlim(FREQ_RANGE)

    # plot 95% confidence intrval
    ax2b.fill_between(freq, conf_pre[0], conf_pre[1], 
                      color=COLORS[f'light_{color}'], alpha=0.5)
    ax2b.fill_between(freq, conf_post[0], conf_post[1], color=COLORS[color], 
                      alpha=0.5)

    # plot intersection frequnency
    ax2b.scatter(f_rotation, np.nanmean(psd_pre_all, 0)[idx_rotation], s=32, 
                 color='k', zorder=5, label='Intersection')

    # subplot 2 - label
    ax2b.set_title('Power Spectra') #('Single-electrode Spectra', fontsize=20)
    ax2b.set_xlabel('frequency (Hz)')
    ax2b.set_ylabel('power ($\u03bcV^2/Hz$)')
    ax2b.legend(markerscale=0.4)
    ax2b.axvline(1, color='gray', linestyle='--', linewidth=1)

    # save figure
    fig.savefig(f"{dir_output}/example_data.png")

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


if __name__ == "__main__":
    main()
