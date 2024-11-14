"""
Simulate the effect of ERPs on spectral power. ERPs were simulated using
the MATLAB toolbox SEREEGA, then added to simulated neural time-series
with different gains. The spectral power of the time-series was then
computed to show the effect of the ERPs on the power spectra.

"""

# Imports - standard
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from neurodsp.utils import create_times
from neurodsp.sim import sim_powerlaw
from neurodsp.spectral import compute_spectrum, rotate_powerlaw

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from utils import get_start_time, print_time_elapsed
from settings import WIDTH, PANEL_FONTSIZE
from plots import beautify_ax, join_two_figures

# settings
plt.style.use('mplstyle/nature_neuro.mplstyle')
N_TRIALS = 100
N_SECONDS = 1 # duration of simulated ERP
FS = 2000 # sampling rate of simulated ERP
GAIN = [0.0, 0.25, 0.5, 0.75] # evoked response gain
EXP_PRE = -2
ROT_FREQ = 30


def main():

    # display progress
    t_start = get_start_time()

    # identify / create directories
    dir_output = f"{PROJECT_PATH}/data/results"
    dir_fig = f"{PROJECT_PATH}/figures/main_figures"
    for dir in [dir_output, dir_fig]:
        if not os.path.exists(dir): 
            os.makedirs(f"{dir}")

    # load simulated ERPs ======================================================
    data_in = loadmat(f"{PROJECT_PATH}/data/simulated_evoked_response/visual_evoked_response.mat")
    evoked_sereega = np.squeeze(data_in['evoked']) * 1E13
    evoked = evoked_sereega / np.sqrt(np.mean(evoked_sereega**2))
        
    # simulate =================================================================
    # simulate pre-stim time-series, average over trials
    prestim = np.zeros([N_TRIALS, N_SECONDS*FS])
    for trial in range(N_TRIALS):
        prestim[trial] = create_powerlaw(N_SECONDS*FS, FS, EXP_PRE, ROT_FREQ)
    prestim_mean = prestim.mean(axis=0)
    signal_pre = np.tile(prestim_mean,(4,1))

    # simulate evoked-response
    results = comp_gain_effect(GAIN, EXP_PRE, evoked)
    _, _, signal_post_evoked, freq_evoked, _, psd_evoked = results

    # concatenate pre and post-stim signals for rotation
    signal_evoked = np.hstack([signal_pre, signal_post_evoked])
    time = create_times(N_SECONDS*2, FS, start_val=-N_SECONDS)     

    # plot =====================================================================
    figsize = [WIDTH['1col'], WIDTH['1col']/2]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize, 
                                   constrained_layout=True)

    # plot evoked response sim
    lw = 1
    ax0.plot(time, signal_evoked[0], color='k', linewidth=lw)
    ax0.plot(time[FS*N_SECONDS:], signal_evoked[1,FS*N_SECONDS:],
              color=[1.0,0.0,0.0], linewidth=lw)
    ax0.plot(time[FS*N_SECONDS:], signal_evoked[2,FS*N_SECONDS:], 
             color=[1.0,0.4,0.0], linewidth=lw)
    ax0.plot(time[FS*N_SECONDS:], signal_evoked[3,FS*N_SECONDS:], 
             color=[1.0,0.8,0.0], linewidth=lw)
    ax0.axvline(0, linestyle='--', color='k', linewidth=lw)
    ax0.set_yticks([0])
    ax0.set_title('Simulated ERP')
    ax0.set(xlabel='time (s)', ylabel='voltage (au)')
    ax0.set_xlim([0, 1])

    # plot power spectra
    lw = 1
    ax1.loglog(freq_evoked, psd_evoked[0], color='k', linewidth=lw)
    ax1.loglog(freq_evoked, psd_evoked[1], color=[1.0,0.0,0.0], linewidth=lw)
    ax1.loglog(freq_evoked, psd_evoked[2], color=[1.0,0.4,0.0], linewidth=lw)
    ax1.loglog(freq_evoked, psd_evoked[3], color=[1.0,0.8,0.0], linewidth=lw)
    ax1.set_xticks([10, 100], labels=['10', '100'])
    ax1.set_title('Power spectra')
    ax1.set(xlabel='frequency (Hz)', ylabel='power (au)')

    # beautify axes
    for ax in [ax0, ax1]:
        beautify_ax(ax)

    # add figure panel labels
    fig.text(0.05, 0.94, 'c', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.55, 0.94, 'd', fontsize=PANEL_FONTSIZE, fontweight='bold')

    # save fig
    fig.savefig(f"{dir_fig}/figure_7-1cd.png")

    # join subplots
    join_two_figures(f"{dir_fig}/figure_7-1ab.png", 
                     f"{dir_fig}/figure_7-1cd.png",
                     f"{dir_fig}/figure_7-1.png",
                     figsize=[WIDTH['1col'], WIDTH['1col']])
    join_two_figures(f"{dir_fig}/figure_7-1ab.png", 
                     f"{dir_fig}/figure_7-1cd.png",
                     f"{dir_fig}/figure_7-1",
                     figsize=[WIDTH['1col'], WIDTH['1col']])

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


def create_powerlaw(n_samples, fs, exponent, rot_freq):
    """Create a power law time series.
    Modified from neurodsp.sim.aperiodic._create_powerlaw.
    Rather than rotating white noise about freq=1Hz, the rotation
    frequency can b specified by the input param rot_freq.
    In addtion, the time-series are not z-scored
    
    Parameters
    ----------
    n_samples : int
        The number of samples to simulate.
    fs : float
        Sampling rate of simulated signal, in Hz.
    exponent : float
        Desired power-law exponent, of the form P(f)=f^exponent.
    Returns
    -------
    sig: 1d array
        Time-series with the desired power law exponent.
    Notes
    -----
    This function creates variable power law exponents by spectrally rotating white noise.
    """

    # Start with white noise signal, that we will rotate, in frequency space
    sig = np.random.randn(n_samples)

    # Compute the FFT
    fft_output = np.fft.fft(sig)
    freqs = np.fft.fftfreq(len(sig), 1. / fs)

    # Rotate spectrum and invert back to time series, with a z-score to normalize
    #   Delta exponent is divided by two, as the FFT output is in units of amplitude not power
    fft_output_rot = rotate_powerlaw(freqs, fft_output, -exponent/2, f_rotation=rot_freq)
    #sig = zscore(np.real(np.fft.ifft(fft_output_rot)))
    sig = np.real(np.fft.ifft(fft_output_rot))

    return sig


def simulate_rotated_spectra(exponents, rot_freq=30, f_range=[1, 300], 
                             n_seconds=300, fs=2000, n_trials=100):

    # simulate neural time-series and compute spectrum
    time = create_times(n_seconds, fs)
    freq = np.arange(f_range[0], f_range[1], 1)
    
    signal = np.zeros([len(exponents), n_trials, n_seconds*fs])
    temp = np.zeros(n_seconds*fs)
    _, temp = compute_spectrum(temp, fs, f_range=f_range)
    psd = np.zeros([len(exponents), n_trials, len(temp)])
    
    for ii, exp in enumerate(exponents):
        for trial in range(n_trials):
            signal[ii, trial] = create_powerlaw(n_seconds*fs, fs, exp, rot_freq)
        freq, psd[ii] = compute_spectrum(signal[ii], fs, f_range=f_range)
            
    # average over trials
    signal = np.squeeze(np.mean(signal, axis=1))
    psd = np.squeeze(np.mean(psd, axis=1))
    
    return time, signal, freq, psd


def comp_spectral_evoked_response(evoked, fs=2000, n_seconds=1, exp=-2, n_trials=100, f_range=[4,100]):
    # simulate neaural time series
    # add evoked responses to trial data
    # compute PSD

    # create time-vector
    time = create_times(n_seconds, fs)

    # create empty arrays for data
    signal = np.zeros([n_trials, len(time)]) 
    signal_plus_evoked = np.zeros([n_trials, len(time)])
    psd = []
    psd_with_evoked = []

    # loop throuhg trials - add evoked response, comp psd
    for ii in range(n_trials):
        # simulate timte-series
        signal[ii] = sim_powerlaw(n_seconds, fs, exp)
        # add evoked reponse
        signal_plus_evoked[ii] = signal[ii] + evoked[ii]

        # calc psd for post-stim window
        freq, psd_ii = compute_spectrum(signal[ii], fs, f_range=f_range)
        freq, psd_with_evoked_ii = compute_spectrum(signal_plus_evoked[ii], fs, 
                                                    f_range=f_range)
        psd.append(psd_ii)
        psd_with_evoked.append(psd_with_evoked_ii)
    
    return time, signal, signal_plus_evoked, freq, psd, psd_with_evoked


def comp_gain_effect(gain, exponent, evoked):
    # create empty arrays for data
    signal = []
    signal_plus_evoked = []
    psd = []
    psd_with_evoked = []

    for ii, g in enumerate(gain):
        results = comp_spectral_evoked_response(evoked*g, exp=exponent,
                                                fs=FS, n_seconds=N_SECONDS,
                                                n_trials=N_TRIALS)
        time, signal_ii, signal_plus_evoked_ii, freq, psd_ii, \
            psd_with_evoked_ii = results
        signal.append(np.mean(signal_ii, axis=0))
        signal_plus_evoked.append(np.mean(signal_plus_evoked_ii, axis=0))
        psd.append(np.mean(psd_ii, axis=0))
        psd_with_evoked.append(np.mean(psd_with_evoked_ii, axis=0))
    
    return time, signal, signal_plus_evoked, freq, psd, psd_with_evoked
        

if __name__ == "__main__":
    main()
