"""
This script plots the "tilt" and "fingerprints" models. Power spectra are
simulated and plotted to illustrate each hypothesis. These models aim to
explain observed increases in high-frequency and decreases in low-frequency
spectral power, respectively, during memory encoding. The "tilt" or "aperiodic"
model simulates a shift in the aperiodic exponent, while the "fingerprints"
or "oscillations" model simulates changes in the periodic/oscillatory component.

"""

# Imports - standard
import os
import numpy as np
import matplotlib.pyplot as plt
from specparam.sim import sim_power_spectrum
from neurodsp.spectral import rotate_powerlaw

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from settings import *

# plotting setting
plt.style.use(MPLSTYLE)

# Set PSD simulation parameters
PARAMS_AP = [5, 2] # aperiodic parameters for simulated power spectra (baseline)
PARAMS_PER = [[10, 1, 1.5], [70, 0.15, 8]] # periodic parameters (baseline)
PARAMS_PER_POST = [[10, 0.25, 1.5], [70, 0.4, 8]] # periodic parameters (encoding)
ROTATION_FREQ = 40 # frequency at which to rotate the spectra
ROTATION_DELTA = -0.75 # change in exponent for rotation
NOISE_LEVEL = 0 # noise level for simulated power spectra
ANNOTATE_ROATATION = True # whether to annotate the rotation frequency


def main():
    # set paths
    path_out = f"{PROJECT_PATH}/figures/main_figures"
    if not os.path.exists(path_out): os.makedirs(path_out)

    # sim baseline spectra (with and without peaks)
    freqs, psd_pre = sim_power_spectrum(FREQ_RANGE, PARAMS_AP, PARAMS_PER, 
                                        nlv=NOISE_LEVEL)
    _, psd_pre_np = sim_power_spectrum(FREQ_RANGE, PARAMS_AP, [], 
                                       nlv=NOISE_LEVEL)

    # simulate oscillatory model
    _, psd_post_0 = sim_power_spectrum(FREQ_RANGE, PARAMS_AP, PARAMS_PER_POST, 
                                       nlv=NOISE_LEVEL)
        
    # simulate aperiodic model - peaks present
    _, psd_post_1 = sim_power_spectrum(FREQ_RANGE, PARAMS_AP, PARAMS_PER, 
                                       nlv=NOISE_LEVEL)
    psd_post_1 = rotate_powerlaw(freqs, psd_post_1, 
                                 delta_exponent=ROTATION_DELTA, 
                                 f_rotation=ROTATION_FREQ)

    # simulate aperiodic model - no peaks present
    psd_post_2 = rotate_powerlaw(freqs, psd_pre_np, 
                                 delta_exponent=ROTATION_DELTA, 
                                 f_rotation=ROTATION_FREQ)

    # simulate combined model 
    psd_post_3 = rotate_powerlaw(freqs, psd_post_0, 
                                 delta_exponent=ROTATION_DELTA, 
                                 f_rotation=ROTATION_FREQ)

    # plot each model
    labels = ['baseline', 'encoding']
    figsize = (WIDTH['1col'], WIDTH['1col'])
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, sharey=True, 
                                               figsize=figsize)
    plot_2_spectra(psd_pre, psd_post_0, freqs, ax=ax0, labels=labels,
                    title='Periodic model')
    plot_2_spectra(psd_pre, psd_post_1, freqs, ax=ax1, labels=labels,
                   title='Aperiodic model')
    plot_2_spectra(psd_pre_np, psd_post_2, freqs, ax=ax2, labels=labels,
                   title='Aperiodic model (no peaks)')
    plot_2_spectra(psd_pre, psd_post_3, freqs, ax=ax3, labels=labels,
                   title='Combined model')

    # shade oscillation bands
    for band in ['alpha', 'gamma']:
        for ax in [ax0, ax1, ax2, ax3]:
            ax.axvspan(BANDS[band][0], BANDS[band][1], facecolor=BCOLORS[band], 
                       alpha=0.4)
            
    # annotate rotation frequency
    if ANNOTATE_ROATATION:
        for ax in [ax1, ax2, ax3]:
            idx_rotation = np.argmin(np.abs(freqs - ROTATION_FREQ))
            ax.scatter(freqs[idx_rotation], psd_post_1[idx_rotation], 
                       color='k', s=15, zorder=10)
            
    # remove crowded y-labels
    for ax in [ax1, ax3]:
        ax.set(ylabel='')

    # add figure panel labels
    fig.text(0.05, 0.97, 'a', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.55, 0.97, 'b', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.05, 0.47, 'c', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.55, 0.47, 'd', fontsize=PANEL_FONTSIZE, fontweight='bold')
            
    # save figure
    plt.savefig(f"{path_out}/figure_1")
    plt.savefig(f"{path_out}/figure_1.png")


def plot_2_spectra(spectrum_0, spectrum_1, freqs, labels=['0', '1'], 
                   title=None, ax=None):
    
    # Create figure
    if ax is None:
        _, ax = plt.subplots()
    
    # plot spectra
    ax.loglog(freqs, spectrum_0, label=labels[0], color='grey')
    ax.loglog(freqs, spectrum_1, label=labels[1], color='k')
    
    # label
    ax.set(xlabel='Frequency (Hz)', ylabel='Power (\u03BCV\u00b2/Hz)')
    ax.set_xticks([1, 10, 100], labels=['0', '10', '100'])
    ax.legend()
    if title is not None:
        ax.set_title(title)


if __name__ == "__main__":
    main()
