"""
Plot intersection frequency results. 

Subplot 0: simulation of 2 spectral rotations (high and low intersection freq.)
Subplot 1: simulation of the effects of spectral rotation on total band power.
Subplot 2: grand average power spectra.
Subplot 3: histogram of intersection frequency.
"""

# Imports - standard
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neurodsp.spectral import rotate_powerlaw
from specparam.sim import sim_power_spectrum
import scipy.stats as stats

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from utils import get_start_time, print_time_elapsed
from info import MATERIALS
from settings import *
from plots import plot_spectra_2conditions, beautify_ax
from specparam_utils import compute_band_power

# settings
plt.style.use('mplstyle/nature_neuro.mplstyle')
figsize = [WIDTH['1col'], WIDTH['1col']*1.5]
IF_SIZE = 10 # intersection frequency marker size

def main():
    # display progress
    t_start = get_start_time()

    # identify / create directories
    dir_output = f"{PROJECT_PATH}/figures/main_figures"
    if not os.path.exists(dir_output): 
        os.makedirs(f"{dir_output}")

    # load task-modulation results
    # df = pd.read_csv(f"{PROJECT_PATH}/data/results/ieeg_modulated_channels.csv")
    fname = f"{PROJECT_PATH}/data/results/band_power_statistics.csv"
    temp = pd.read_csv(fname, index_col=0)
    df_w = temp.loc[(temp['material']=='words') & (temp['memory']=='hit')]
    df_f = temp.loc[(temp['material']=='faces') & (temp['memory']=='hit')]
    for df in [df_w, df_f]: # compute joint significance
        df.insert(0, 'sig_any', df.get([f'{band}_sig' for band in BANDS]).any(axis=1))
        df.insert(0, 'sig_all', df.get([f'{band}_sig' for band in BANDS]).all(axis=1))

    # load rotation analysis results
    intersection = dict()
    for df, material in zip([df_w, df_f], MATERIALS):
        fname = f"intersection_results_{material}_hit_knee.npz"
        data_in = np.load(f"{PROJECT_PATH}/data/ieeg_intersection_results/{fname}")
        intersection[material] = data_in['intersection'][df['sig_all']]
    
    # load spectal results
    psd = dict()
    for material in MATERIALS:
        for epoch in ['pre','post']:
            fname = f"psd_{material}_hit_{epoch}stim.npz"
            data_in = np.load(f"{PROJECT_PATH}/data/ieeg_spectral_results/{fname}")
            psd[(material, epoch)] = data_in['spectra'][df['sig_all']]
    freq = data_in['freq']

    # compute grand average spectra
    spectra_pre = dict()
    spectra_post = dict()
    for material in MATERIALS:
        spectra_pre[material] = psd[material, 'pre']
        spectra_post[material] = psd[material, 'post']

    # plot =====================================================================

    # create figure
    fig, axes = plt.subplots(3, 2, figsize=figsize, constrained_layout=True)
    ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = axes

    # plot simulation: Effects of spectral rotation on total band power
    simulation_subplot_0(fig, ax1)
    simulation_subplot_1(ax2)

    # plot empirical spectra
    for material, ax, color in zip(MATERIALS, [ax3, ax5], ['brown', 'blue']):
        # plot spectra
        colors = [COLORS[f'light_{color}'], COLORS[color]]
        idx_fit = np.logical_and(freq >= FREQ_RANGE[0], freq <= FREQ_RANGE[1])
        plot_spectra_2conditions(spectra_pre[material][:, idx_fit], 
                                 spectra_post[material][:, idx_fit], 
                                 freq[idx_fit], ax=ax, shade_sem=False,
                                 color=colors)
        ax.grid(False)

        # annotate intersection frequency on power spectra
        f_intersection = intersection[material]
        median_f = np.nanmedian(f_intersection)
        median_idx = np.argmin(np.abs(freq - median_f))
        y_val = np.nanmean(spectra_pre[material][:, median_idx])
        ax.scatter(freq[median_idx], y_val, color=BCOLORS['exponent'], 
                   s=IF_SIZE, zorder=10)
        ax.legend(['baseline', 'encoding', 'intersection'], loc='lower left')

    # plot histogram of intersection frequency
    bins = np.linspace(0, 100, 14) # 13 bins based on Freedman-Diaconis rule
    for material, ax in zip(MATERIALS, [ax4, ax6]):
        ax.hist(intersection[material], bins, color=BCOLORS['exponent'], 
                edgecolor='k', linewidth=0.5)
        ax.set_xlabel('frequency (Hz)')
        ax.set_ylabel('electrode count')
    
    # remove top and right spines
    for ax in np.ravel(axes):
        beautify_ax(ax)

    # set titles
    ax1.set_title('Rotations at high and low\nintersection frequency (IF)')
    ax2.set_title('Effect of intersection frequency\non total band power')
    for ax in [ax3, ax5]:
        ax.set_title('\n\nAverage power spectra')
    for ax in [ax4, ax6]:
        ax.set_title('\n\nIntersection frequency')

    # set xlimits to match across columns
    for ax in [ax2, ax4, ax6]:
        ax.set_xlim([0, 100])

    # add text above each row of subplots
    for title, ypos in zip(['Simulation', 'Empirical: word-encoding', 
                            'Empirical: face-encoding'], [1.02, 0.66, 0.32]):
        fig.text(0.5, ypos, title, ha='center', va='center', fontsize=7, 
                fontweight='bold')

    # add figure panel labels
    fig.text(0.05, 0.97, 'a', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.53, 0.97, 'b', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.05, 0.62, 'c', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.53, 0.62, 'd', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.05, 0.28, 'e', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.53, 0.28, 'f', fontsize=PANEL_FONTSIZE, fontweight='bold')

    # save figure
    fig.savefig(f"{dir_output}/figure_5", bbox_inches='tight')
    fig.savefig(f"{dir_output}/figure_5.png", bbox_inches='tight')

    # print average intersection frequency and standard deviation
    for material in MATERIALS:
        f_intersection = intersection[material]
        print(f"\n\n{material}-encoding - intersection frequency:")
        print(f"    Mean:\t{np.nanmean(f_intersection):.3f} Hz")
        print(f"    Median:\t{np.nanmedian(f_intersection):.3f} Hz")
        print(f"    STD:\t{np.nanstd(f_intersection):.3f} Hz")

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


def simulation_subplot_0(fig, ax):
    """ 
    Simulate a power spectrum and rotate it at a high and a low intersection
    frequency. Plot the original and rotated power spectra.
    """

    # simulate power spectrum and rotate
    freqs, psd_pre = sim_power_spectrum(freq_range=FREQ_RANGE, 
                                        aperiodic_params=[10, 2], 
                                        periodic_params=[])
    psd_rot_low = rotate_powerlaw(freqs, psd_pre, delta_exponent=-0.5, 
                                  f_rotation=FREQ_RANGE[0])
    psd_rot_high = rotate_powerlaw(freqs, psd_pre, delta_exponent=-0.5, 
                                  f_rotation=FREQ_RANGE[1])
    
    # plot power spectra
    ax.loglog(freqs, psd_pre, color='k', label='baseline')
    ax.loglog(freqs, psd_rot_high, color=RGB[2], linestyle='--',
              label='high IF')
    ax.loglog(freqs, psd_rot_low, color=RGB[0], linestyle='--',
              label='low IF')
    
    # plot intersection frequency
    ax.scatter(FREQ_RANGE[0], psd_rot_low[0], color=RGB[0], s=IF_SIZE, 
               zorder=10)
    ax.scatter(FREQ_RANGE[1], psd_rot_high[-1], color=RGB[2], s=IF_SIZE, 
               zorder=10)

    # shade region between original and rotated spectra
    ax.fill_between(freqs, psd_pre, psd_rot_high, color=RGB[2], alpha=0.3)
    ax.fill_between(freqs, psd_pre, psd_rot_low, color=RGB[0], alpha=0.3)
    fig.text(0.20, 0.88, 'broadband decrease', ha='center', va='center', 
             rotation=-40, fontsize=5)
    fig.text(0.39, 0.805, 'broadband increase', ha='center', va='center', 
             rotation=-40, fontsize=5)

    # label plot
    ax.set(xlabel='frequency (Hz)', ylabel='power (au)')
    ax.set_xticks([10, 100], labels=['10', '100'])
    ax.legend()

    
def simulation_subplot_1(ax):
    """ 
    Run simulation and plot results. Effects of spectral rotation on total band 
    power.
    """

    # run simulation with different rotation frequencies
    f_rotations = np.arange(5, 105, 5)
    dfs = []
    for f_rotation in f_rotations:
        df = run_simulation(f_rotation, f_range=FREQ_RANGE)
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
                    f_range=[1, 100]):
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
