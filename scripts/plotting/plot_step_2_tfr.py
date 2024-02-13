# -*- coding: utf-8 -*-
"""
This script plots the TFR results of code/2_time_frequency_analysis.py

"""

# Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from time import time as timer

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from utils import hour_min_sec
from plots import plot_tfr, plot_spectra_2conditions
from tfr_utils import load_tfr_results, crop_tfr

# Settings
WINDOW = 0.3 # in seconds. Edge effects are removed by cropping the TFR results
T_BASELINE = [-1, 0] # in seconds. These time bins will be averaged for PSD plot
T_ENCODING = [0, 1] # in seconds. These time bins will be averaged for PSD plot
T_SPECTROGRAM = [-1, 2] # in seconds. Spectrogram will be cropped in this range for plotting

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

def main():
    # time it
    t_start = timer()

    # identify / create directories
    dir_input = f"{PROJECT_PATH}/data/ieeg_tfr"
    dir_output = f"{PROJECT_PATH}/figures/spectrograms"
    if not os.path.exists(dir_output): 
        os.makedirs(dir_output)

    # load task-modulation results
    df_tm = pd.read_csv(f"{PROJECT_PATH}/data/results/ieeg_modulated_channels.csv")
    df = df_tm.loc[(df_tm['material']=='faces') & (df_tm['memory']=='hit')]
    df = df.drop(columns=['material','memory'])
    df = df.loc[df['sig']].reset_index(drop=True)
    
    # loop through significant channels
    for i_chan, row in df.iterrows():
        # display progress
        t_start_r = timer()
        print(f"\nPlotting channel {i_chan+1}/{len(df)}")
        print(f"    Patient ID: \t{row['patient']}")
        print(f"    Channel idx: \t{row['chan_idx']}\n")
        print(f"    Plotting each condition:")
        
        # loop through conditions
        for material in ['word','face']:
            for memory in ['hit','miss']:
                # file name for input/output
                fname = f"{row['patient']}_{material}s_{memory}_chan{row['chan_idx']}"
                
                # display progress
                print(f"\t{material}-{memory}...")
                
                # load spectral results
                fname_in = f"{dir_input}/{fname}_tfr.npz"
                time, freq, tfr = load_tfr_results(fname_in, preprocess=True, edge=WINDOW, t_baseline=None, z_score=False)
                _, _, tfr_norm = load_tfr_results(fname_in, preprocess=True, edge=WINDOW, t_baseline='default')

                # check data exists
                if tfr is None or np.isnan(tfr).all():
                    print(f"\t\tNo data for this condition. Skipping...")
                    continue

                # create figure
                fig, (ax_1, ax_2) = plt.subplots(1,2, figsize=[10,4], constrained_layout=True)

                # plot spectrogram
                tfr_plot, time_plot = crop_tfr(tfr_norm, time, T_SPECTROGRAM)
                plot_tfr(time_plot, freq, tfr_plot, norm_type='centered', cbar_label='normalizaed power', 
                         title='Normalized spectrogram', annotate_zero=True, fig=fig, ax=ax_1)

                # Plot average spectra for pre- and post-stimulus periods
                spectra_pre = tfr[:, ((time>T_BASELINE[0]) & (time<T_BASELINE[1]))].T
                spectra_post = tfr[:, ((time>T_ENCODING[0]) & (time<T_ENCODING[1]))].T
                plot_spectra_2conditions(spectra_pre, spectra_post, freq, shade_sem=True, ax=ax_2,
                                         title='Average spectral power')

                # save figure
                fig.suptitle(f"{row['patient']}, chan{row['chan_idx']}: {material}-{memory}")
                fig.savefig(f"{dir_output}/{fname}.png")

        # display progress
        hour, min, sec = hour_min_sec(timer() - t_start_r)
        print(f"\n    Channel complete in: {hour} hour, {min} min, and {sec :0.1f} s")

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\nTotal time: {hour} hour, {min} min, and {sec :0.1f} s")


if __name__ == "__main__":
    main()