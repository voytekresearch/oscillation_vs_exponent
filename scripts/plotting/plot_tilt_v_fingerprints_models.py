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
import matplotlib.pyplot as plt
from specparam.sim import sim_power_spectrum
from neurodsp.spectral import rotate_powerlaw

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH

# plotting setting
plt.style.use('mpl_styles/default.mplstyle')
cols = ['k', 'blue'] # ['w','gold']

# Set PSD simulation parameters
freq_range = [2,100]
params_ap = [10, 3.5] 


def main():
    # set paths
    path_out = f"{PROJECT_PATH}/figures/model_sketch"
    if not os.path.exists(path_out): os.makedirs(path_out)

    # Fingerprints model simulation ==========================================
    # sim baseline spectra
    params_per = [[15,1.5,3],[70,0.5,8]]
    freqs, spectrum_pre = sim_power_spectrum(freq_range, params_ap, params_per)

    # sim encoding spectra
    params_per = [[15,0.5,3],[70,1,8]]
    freqs, spectrum_post = sim_power_spectrum(freq_range, params_ap, params_per)

    # plot
    fname_out = f'{path_out}/fingerprint_model_sim'
    title = 'Model: Oscillations' # 'Model: Spectral Fingerprints'
    plot_model_sketch(spectrum_pre, spectrum_post, freqs, title, fname_out)

    # tilt model simulation ================================================
    # plot spectral tild model (with and without peaks)
    fname_out = f'{path_out}/tilt_model_sim'
    for params_per, tag in zip([[[15,1.5,3],[70,0.5,8]],[]], ['', '_no_peaks']):
                                    
        # sim baseline spectra
        freqs, spectrum_pre = sim_power_spectrum(freq_range, params_ap, 
                                                 params_per)
        
        # sim encoding spectra - using neurodsp function
        freqs, spectrum_post = sim_power_spectrum(freq_range, params_ap, 
                                                  params_per)
        spectrum_post = rotate_powerlaw(freqs, spectrum_post, 
                                        delta_exponent=-0.75, f_rotation=42)
        
        # plot
        title = 'Model: Aperiodic Shift' # 'Model: Rotation'
        plot_model_sketch(spectrum_pre, spectrum_post, freqs, title, 
                          f"{fname_out}{tag}")

    # Mixed model prediction ================================================
    # sim baseline spectra
    params_per = [15,1.5,3]
    freqs, spectrum_pre = sim_power_spectrum(freq_range, params_ap, params_per)

    # sim encoding spectra
    params_per = [15,0.5,3]
    freqs, spectrum_post = sim_power_spectrum(freq_range, params_ap, params_per)
    spectrum_post = rotate_powerlaw(freqs, spectrum_post, 
                                    delta_exponent=-0.75, f_rotation=42)

    # plot
    plot_model_sketch(spectrum_pre, spectrum_post, freqs, 
                    'Model: Mixed', 'mixed_model')


def plot_model_sketch(spectrum_pre, spectrum_post, freqs, title, fname_out):
    # Create figure
    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot()
    
    # plot spectra
    plt.loglog(freqs, spectrum_pre, label='baseline', color=cols[0], 
               linewidth=3)
    plt.loglog(freqs, spectrum_post, label='encoding', color=cols[1], 
               linewidth=3)
    
    # label
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (\u03BCV\u00b2/Hz)')
    plt.legend()
    plt.xticks([1,10,100],['0','10','100'])
    
    # annotations
    #ax.axvspan(2, 5, facecolor='grey', alpha=0.4)
    ax.axvspan(8, 20, facecolor='grey', alpha=0.4)
    ax.axvspan(50, 90, facecolor='grey', alpha=0.4)
#    ax.text(0.38, 0.925, 'Alpha / Beta', fontsize=10, color=cols[0], 
#            transform=ax.transAxes)
#    ax.text(0.805, 0.925, 'Gamma', fontsize=10, color=cols[0], 
#            transform=ax.transAxes)
    ax.text(0.38, 0.925, 'Low', fontsize=10, color=cols[0], 
            transform=ax.transAxes)
    ax.text(0.805, 0.925, 'High', fontsize=10, color=cols[0], 
            transform=ax.transAxes)
        
    # save and close 
    plt.savefig(fname_out)
    

if __name__ == "__main__":
    main()
