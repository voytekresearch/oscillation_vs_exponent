"""
Plot example data for two electrodes: one with a clear ERP and no exponent 
modulation and another with clear exponent modulation and no ERP.
"""

# Imports - standard
import os
import numpy as np
import matplotlib.pyplot as plt
from mne import read_epochs
from specparam import SpectralGroupModel, fit_models_3d

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from utils import get_start_time, print_time_elapsed, confidence_interval
from erp_utils import subtract_baseline
from tfr_utils import crop_tfr
from settings import *
from info import TMIN
from plots import beautify_ax

# settings - example data
PATIENT = ['pat19', 'pat05']
CHAN_IDX = [66, 48]
MATERIAL = ['faces', 'faces']
MEMORY = ['miss', 'hit']

# settings
plt.style.use(MPLSTYLE)
T_PLOT = [-0.5, 1.] # time window to plot
LOAD_DATA = True # whether to load data or compute


def main():

    # display progress
    t_start = get_start_time()

    # identify / create directories
    dir_output = f"{PROJECT_PATH}/data/results"
    dir_fig = f"{PROJECT_PATH}/figures/main_figures"
    for dir in [dir_output, dir_fig]:
        if not os.path.exists(dir): 
            os.makedirs(f"{dir}")
        
    # compute ERP ==============================================================
    erp_list = []

    # loop through conditions
    for ii in range(2):
        # load epochs
        fname = f"{PATIENT[ii]}_{MATERIAL[ii]}_{MEMORY[ii]}_epo.fif"
        epochs = read_epochs(f"{PROJECT_PATH}/data/ieeg_epochs/{fname}")
  
        # get event traces
        signals = epochs.get_data()[:, CHAN_IDX[ii]]
        erp_time = epochs.times

        # remove nan trials
        signals = signals[~np.isnan(signals).any(axis=1)]

        # subtract baseline
        for i_trial in range(signals.shape[0]):
            signals[i_trial] = subtract_baseline(signals[i_trial], erp_time, 
                                                 [TMIN, 0])
            
        # trim signal
        signals, erp_time = crop_tfr(signals, erp_time, T_PLOT)

        erp_list.append(signals)

    # compute exp ==============================================================
    exp_list = []

    for ii in range(2):
        print(f"\nCondition {ii+1} of 2")
        
        if LOAD_DATA:
            # load pre-computed exp time-series
            exp_list.append(np.load(f"{dir_output}/exp_ts_{ii}.npy"))
            
            # load time vector
            if ii == 0:
                fname = f"{PATIENT[ii]}_{MATERIAL[ii]}_{MEMORY[ii]}_chan{CHAN_IDX[ii]}_tfr.npz"
                data_in = np.load(f"{PROJECT_PATH}/data/ieeg_tfr/{fname}", 
                                allow_pickle=True)
                exp_time = data_in['time']

        else:
            # load tfr
            fname = f"{PATIENT[ii]}_{MATERIAL[ii]}_{MEMORY[ii]}_chan{CHAN_IDX[ii]}_tfr.npz"
            data_in = np.load(f"{PROJECT_PATH}/data/ieeg_tfr/{fname}", 
                            allow_pickle=True)
            tfr = np.swapaxes(data_in['tfr'], 1, 2) # swap axes for model fitting
            tfr = tfr[~np.isnan(tfr).all(axis=(1,2))] # remove all nan trials
            exp_time = data_in['time']

            # parameterize
            sgm = SpectralGroupModel(**SPEC_PARAM_SETTINGS)
            sgms = fit_models_3d(sgm, data_in['freq'], tfr, freq_range=FREQ_RANGE, 
                                n_jobs=N_JOBS)
            
            # extract exponent and subtract baseline
            exp_list_i = []
            for sgm in sgms:
                exp_j = sgm.get_params('aperiodic', 'exponent')
                exp_j = subtract_baseline(exp_j, exp_time, [TMIN, 0])
                exp_list_i.append(exp_j)
            exp_list.append(np.array(exp_list_i))
            np.save(f"{dir_output}/exp_ts_{ii}.npy", exp_list[ii])


    # plot =====================================================================
    figsize = [WIDTH['1col'], WIDTH['1col']/2]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize, sharex=True,
                                    constrained_layout=True)

    # plot data
    for ii, (label, color) in enumerate(zip(['electrode 1', 'electrode 2'],
                                          [RGB[1], RGB[2]])):
        for var_list, time, ax in zip([erp_list, exp_list], 
                                        [erp_time, exp_time], 
                                        [ax0, ax1]):
            ci = confidence_interval(var_list[ii])
            ax.plot(time, np.nanmean(var_list[ii], axis=0), color=color,
                    label=label)
            ax.fill_between(time, ci[0], ci[1], color=color, alpha=0.2)

    # label / format 
    ax0.set(xlabel=f'time (s)', ylabel=f'voltage (\u03BCV)')
    ax0.set_title('Event-related potential')

    ax1.set(xlabel=f'time (s)', ylabel=f'exponent')
    ax1.set_title('Aperiodic exponent')

    for ax in [ax0, ax1]:
        ax.axvline(0, color='k', linestyle='--')
        ax.axhline(0, color='k', linestyle='--')
        ax.set_xlim(T_PLOT)
        ax.legend(loc='lower left')

    # beautify axes
    for ax in [ax0, ax1]:
        beautify_ax(ax)

    # add figure panel labels
    fig.text(0.05, 0.94, 'a', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.55, 0.94, 'b', fontsize=PANEL_FONTSIZE, fontweight='bold')

    # save fig
    fig.savefig(f"{dir_fig}/figure_7-2ab.png")

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


if __name__ == "__main__":
    main()
