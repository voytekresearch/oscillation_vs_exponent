# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 17:35:00 2021

@author: micha

Data Repo: https://osf.io/3csku/
Associated Paper: https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000403

 
This script executes the primary time-frequnecy analyses. 
The power spectral density (PSD) is computed for each epoch, the pre-stimulus 
time window, and the post-stimulus time window. 
The time-frequnecy representation of power (TFR) is computed for the epoch 
using the multitaper method. 

"""
#  SET PATH
PROJECT_PATH = 'C:/Users/micha/projects/oscillation_vs_exponent/'

# settings 
T_BASELINE = [-0.5, 0.] # baseline time window for ERP computation
T_TRIM = [-0.5, 2.] # time window to trim signal

# Imports - general
from os.path import join, exists
from os import mkdir, listdir
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mne import read_epochs
from time import time as timer

# Imports - custom
import sys
sys.path.append(f"{PROJECT_PATH}/code")
from utils import hour_min_sec
from tfr_utils import crop_tfr
from erp_utils import compute_erp, plot_erp

# Dataset details
PATIENTS = ['pat02','pat04','pat05','pat08','pat10','pat11','pat15','pat16',
            'pat17','pat19','pat20','pat21','pat22']

# set plotting parameers
mpl.rcParams['figure.facecolor'] = 'w'
mpl.rcParams['axes.facecolor'] = 'w'
mpl.rcParams['figure.titlesize'] = 18
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['font.size'] = 10
mpl.rcParams['figure.constrained_layout.use'] = True

def main():
    # display progress
    t_start = timer()

    # identify / create directories
    dir_input = join(PROJECT_PATH, 'data/ieeg_epochs')
    dir_output = join(PROJECT_PATH, 'data/results')
    dir_fig = join(PROJECT_PATH, 'figures/erp')
    if not exists(dir_output): mkdir(dir_output)
    if not exists(dir_fig): mkdir(dir_fig)
    
    # init
    df_erp = pd.DataFrame(columns=['patient', 'chan_idx', 'material', 'memory', 'erp_amp'])

    # for each fif file
    files = listdir(dir_input)
    for ii, fname in enumerate(files):

        # display progress
        t_start_f = timer()
        print(f"\nAnalyzing file {ii+1}/{len(files)}")
        print(f"\tfilename: \t{fname}")

        # load eeg data
        epochs = read_epochs(join(dir_input, fname), verbose=False)
        signals = epochs.get_data()
        time = epochs.times
        print(f"\tchannels: \t{len(epochs.info['ch_names'])}")
    
        # compute erp
        erps = compute_erp(signals, time, T_BASELINE)

        # trim signal
        erps, time = crop_tfr(erps, time, T_TRIM)

        # compute ERP amplitude and add to df
        erps_abs = np.abs(erps)
        erp_max_pre = np.nanmax(erps_abs[:, time<0], axis=1)
        erp_max_post = np.nanmax(erps_abs[:, time>0], axis=1)
        erp_amp = erp_max_post / erp_max_pre
        fparts = fname.split('_')
        df_i = pd.DataFrame({'patient': fparts[0],
                            'chan_idx': np.arange(erps.shape[0]),
                            'material': fparts[1],
                            'memory': fparts[2],
                            'erp_max_pre': erp_max_pre,
                            'erp_max_post': erp_max_post,
                            'erp_amp': erp_amp})
        df_erp = pd.concat([df_erp, df_i], axis=0)  

        # plot erp for each channel
        for channel in range(erps.shape[0]):
            # plot
            plot_erp(erps[channel], time)

            # set title
            ax = plt.gca()
            ax.set_title(fname)

            # save figure
            fig = plt.gcf()
            fig.set_size_inches(5,4)
            fname_fig = fname.replace('.fif', f'_chan{channel}.png')
            fig.savefig(join(dir_fig, fname_fig))
            plt.close('all')

        # display progress
        hour, min, sec = hour_min_sec(timer() - t_start_f)
        print(f"\tanalysis time: \t{hour} hour, {min} min, and {sec :0.1f} s")

    # save df
    df_erp.to_csv(join(dir_output, 'df_erp.csv'), index=False)

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\n\nTotal analysis time: {hour} hour, {min} min, and {sec :0.1f} s")
    

if __name__ == "__main__":
    main()
    
    