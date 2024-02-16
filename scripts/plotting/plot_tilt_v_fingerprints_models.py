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
from info import FELLNER_BANDS as BANDS

# plotting setting
plt.style.use('mpl_styles/default.mplstyle')
FIG_SIZE = (7.5, 2)
COLORS = (np.array([1,133,113])/255, np.array([166,97,26])/255) # PSD colors
ALPHA = 0.8 # PSD line transparency

# Set PSD simulation parameters
FREQ_RANGE = [1, 100] # frequency range for simulated power spectra
PARAMS_AP = [5, 2] # aperiodic parameters for simulated power spectra (baseline)
PARAMS_PER = [[12, 1, 2], [70, 0.15, 8]] # periodic parameters (baseline)
PARAMS_PER_POST = [[12, 0.25, 2], [70, 0.4, 8]] # periodic parameters (encoding)
ROTATION_FREQ = 40 # frequency at which to rotate the spectra
ROTATION_DELTA = -0.75 # change in exponent for rotation
NOISE_LEVEL = 0 # noise level for simulated power spectra
ANNOTATE_ROATATION = True # whether to annotate the rotation frequency

def main():
    # set paths
    path_out = f"{PROJECT_PATH}/figures/model_sketch"
    if not os.path.exists(path_out): os.makedirs(path_out)

    # sim baseline spectra
    freqs, psd_pre = sim_power_spectrum(FREQ_RANGE, PARAMS_AP, PARAMS_PER, 
                                        nlv=NOISE_LEVEL)

    # simulate oscillatory change
    _, psd_post_0 = sim_power_spectrum(FREQ_RANGE, PARAMS_AP, PARAMS_PER_POST, 
                                       nlv=NOISE_LEVEL)
        
    # sim aperiodic shift
    _, psd_post_1 = sim_power_spectrum(FREQ_RANGE, PARAMS_AP, PARAMS_PER, 
                                       nlv=NOISE_LEVEL)
    psd_post_1 = rotate_powerlaw(freqs, psd_post_1, 
                                 delta_exponent=ROTATION_DELTA, 
                                 f_rotation=ROTATION_FREQ)

    # sim combined model 
    psd_post_2 = rotate_powerlaw(freqs, psd_post_0, 
                                 delta_exponent=ROTATION_DELTA, 
                                 f_rotation=ROTATION_FREQ)

    # plot each model and shade oscillation bands
    _, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=FIG_SIZE)
    plot_2_spectra(psd_pre, psd_post_0, freqs, ax=ax0, colors=COLORS,
                    alpha=ALPHA, labels=['baseline', 'encoding'],
                    title='Oscillatory Amplitude Change')
    plot_2_spectra(psd_pre, psd_post_1, freqs, ax=ax1, colors=COLORS,
                    alpha=ALPHA, labels=['baseline', 'encoding'],
                    title='Aperiodic Exponent Shift')
    plot_2_spectra(psd_pre, psd_post_2, freqs, ax=ax2, colors=COLORS,
                    alpha=ALPHA, labels=['baseline', 'encoding'], 
                    title='Combined Effect')
    
    # shade oscillation bands
    for band in ['alpha', 'gamma']:
        for ax in [ax0, ax1, ax2]:
            ax.axvspan(BANDS[band][0], BANDS[band][1], facecolor='grey', 
                       alpha=0.4)
            
    # annotate rotation frequency
    if ANNOTATE_ROATATION:
        for ax in [ax1, ax2]:
            idx_rotation = np.argmin(np.abs(freqs - ROTATION_FREQ))
            ax.scatter(freqs[idx_rotation], psd_post_1[idx_rotation], color='k', 
                       s=15, zorder=10)
            
    # remove crowded y-labels
    for ax in [ax1, ax2]:
        ax.set(ylabel='')
            
    # save figure
    plt.savefig(f"{path_out}/tilt_v_fingerprints_models.png", dpi=300)


def plot_2_spectra(spectrum_0, spectrum_1, freqs, labels=['0', '1'], 
                   colors=['grey', 'k'], alpha=1, title=None, ax=None):
    
    # Create figure
    if ax is None:
        _, ax = plt.subplots()
    
    # plot spectra
    ax.loglog(freqs, spectrum_0, label=labels[0], color=colors[0], alpha=alpha)
    ax.loglog(freqs, spectrum_1, label=labels[1], color=colors[1], alpha=alpha)
    
    # label
    ax.set(xlabel='Frequency (Hz)', ylabel='Power (\u03BCV\u00b2/Hz)')
    ax.set_xticks([1, 10, 100], labels=['0', '10', '100'])
    ax.legend()
    if title is not None:
        ax.set_title(title)


if __name__ == "__main__":
    main()
