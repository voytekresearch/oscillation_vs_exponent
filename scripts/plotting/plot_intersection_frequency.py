"""
Plot intersection frequency results. 

Subplot A: simulation of the effects of spectral rotation on total band power.
Subplot B: grand average power spectra.
Subplot C: histogram of intersection frequency.
"""

# Imports - standard
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neurodsp.spectral import rotate_powerlaw
from specparam.sim import sim_power_spectrum

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from utils import get_start_time, print_time_elapsed
from info import MATERIALS
from settings import FREQ_RANGE, RGB, WIDTH, BANDS, BCOLORS
from plots import plot_spectra_2conditions, beautify_ax
from specparam_utils import compute_band_power

# settings
plt.style.use('mplstyle/default.mplstyle')
figsize = [WIDTH['2col'], WIDTH['2col']/3]


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
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=figsize, 
                                        constrained_layout=True)

    # plot simulation: Effects of spectral rotation on total band power
    simulation_subplot(ax0)

    # plot empirical spectra
    plot_spectra_2conditions(spectra_pre, spectra_post, freq, ax=ax1,
                                color=['grey', 'k'],
                                shade_sem=False)
    ax1.set_xlim(FREQ_RANGE)
    ax1.grid(False)

    # annotate intersection frequency
    median_f = np.nanmedian(f_intersection)
    median_idx = np.argmin(np.abs(freq - median_f))
    ax1.scatter(freq[median_idx], np.nanmean(spectra_pre[:, median_idx]),
                color=RGB[2], s=20, zorder=10)
    ax1.legend(['baseline', 'encoding', 'intersection'], loc='lower left')

    # plot histogram of intersection frequency
    bins = np.linspace(0, 100, 11)
    ax2.hist(f_intersection, bins, color=RGB[2])
    ax2.set_xlabel('frequency (Hz)')
    ax2.set_ylabel('electrode count')

    # remove top and right spines
    for ax in [ax1, ax2]:
        beautify_ax(ax)

    # set titles
    ax0.set_title('Simulation:\neffects of spectral rotation')
    ax1.set_title('Grand average power spectra')
    ax2.set_title('Intersection frequency')

    # save fig
    fig.savefig(f"{dir_output}/intersection_frequency")
    fig.savefig(f"{dir_output}/intersection_frequency.png")

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


def simulation_subplot(ax):
    """ 
    Run simulation and plot results.
    """

    # run simulation with different rotation frequencies
    f_rotations = np.arange(5, 105, 5)
    dfs = []
    for f_rotation in f_rotations:
        df = run_simulation(f_rotation)
        df['f_rotation'] = f_rotation
        dfs.append(df)
    results = pd.concat(dfs, axis=0, ignore_index=True)

    # plot change in power v rotation frequency
    for band, _ in BANDS.items():
        df = results.loc[results['band']==band]
        df.plot(x='f_rotation', y='diff', ax=ax, color=BCOLORS[band])

    # label plot
    ax.set(xlabel='intersection frequency (Hz)', ylabel='$\Delta$ total band power')
    ax.axhline(0, color='k')
    handles, labels = ax.get_legend_handles_labels()
    labels = [band for band, _ in BANDS.items()]
    ax.legend(handles, labels)

    # annotate each bands
    for band, f_range in BANDS.items():
        # shade band
        ax.axvspan(f_range[0], f_range[1], color=BCOLORS[band], alpha=0.2)

        # mark center
        ax.axvline(np.mean(f_range), linestyle='--', color='k', linewidth=1)


def run_simulation(f_rotation, delta_exponent=1, aperiodic_params=[10, 2],
                    f_range=[4, 100]):
    """ 
    Simulate power spectrum and rotate it. Compute band power before and 
    after rotation.
    """

    # simulate power spectrum and rotate
    freqs, psd_pre = sim_power_spectrum(f_range, aperiodic_params, [])
    psd_post = rotate_powerlaw(freqs, psd_pre, delta_exponent, f_rotation)
    spectra = np.vstack([psd_pre, psd_post])

    # compute band power
    df_list = []
    for band, f_range in BANDS.items():
        power = compute_band_power(freqs, spectra, f_range, method='mean', log_power=True)
        for i_epoch, epoch in enumerate(['pre', 'post']):
            df_i = pd.DataFrame({'epoch': epoch, 'band': band, 
                                'power': power[i_epoch]}, index=[0])
            df_list.append(df_i)
    df = pd.concat(df_list, axis=0)

    # compute change in power
    results = df.pivot(index='band', columns='epoch', values='power').reset_index()
    results['diff'] = results['post'] - results['pre']

    return results


if __name__ == "__main__":
    main()
