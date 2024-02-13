# -*- coding: utf-8 -*-
"""
This script plots the PSD results of code/2_time_frequency_analysis.py

"""

# Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time as timer

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from utils import hour_min_sec, diff_spectra
from plots import plot_spectra_2conditions, plot_psd_diff

# set plotting parameers
plt.style.use('mpl_styles/default.mplstyle')

def main():
    # time it
    t_start = timer()

    # identify / create directories
    dir_input = f"{PROJECT_PATH}/data/ieeg_psd"
    dir_output = f"{PROJECT_PATH}/figures/spectra"
    if not os.path.exists(dir_output): os.makedirs(dir_output)

    # load task-modulation results
    df_tm = pd.read_csv(f"{PROJECT_PATH}/data/results/ieeg_modulated_channels.csv")
    df = df_tm.loc[(df_tm['material']=='faces') & (df_tm['memory']=='hit')]
    df = df.drop(columns=['material','memory'])
    df = df.loc[df['sig']].reset_index(drop=True)
    
    # loop through patient with significant channels
    patients_sig = df['patient'].unique()
    for i_pat, patient in enumerate(patients_sig):
        # get patient data
        df_patient = df.loc[df['patient']==patient].reset_index(drop=True)

        # display progress
        t_start_p = timer()
        print(f"\nPlotting patient {i_pat+1}/{len(patients_sig)}")
        print(f"    Patient ID: \t\t{patient}")
        print(f"    Significant channels: \t{len(df_patient)}\n")
        print(f"    Plotting each condition:")
        
        # loop through conditions

        for material in ['word','face']:
            for memory in ['hit','miss']:
                fname = f"{patient}_{material}s_{memory}_XXXstim_psd.npz"
                
                # display progress
                print(f"\t{material}-{memory}...")
                
                # load spectral results
                data_pre = np.load(f"{dir_input}/{fname.replace('XXX','pre')}")
                data_post = np.load(f"{dir_input}/{fname.replace('XXX','post')}")
                freq = data_pre['freq']

                # loop through significant channels for this patient
                for i_chan in range(len(df_patient)):
                    # set output filename
                    chan_idx = df_patient.loc[i_chan, 'chan_idx']
                    fname_out = f"{patient}_chan{chan_idx}_{material}_{memory}"

                    # get channel data
                    psd_pre = data_pre['psd'][:,chan_idx]
                    psd_post = data_post['psd'][:,chan_idx]

                    # check data exists
                    if np.isnan(psd_pre).all() or np.isnan(psd_post).all():
                        continue
                    if (psd_pre==0).all() or (psd_post==0).all():
                        continue

                    # create figure
                    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4), constrained_layout=True)
                    fig.suptitle(f"{patient}, chan{chan_idx}: {material}-{memory}")

                    # plot pre v post spectra
                    plot_spectra_2conditions(psd_pre, psd_post, freq, ax=ax1)

                    # plot spectral differenct
                    psd_diff = diff_spectra(psd_pre, psd_post)
                    plot_psd_diff(freq, psd_diff, ax=ax2)

                    # save figure
                    fig.savefig(f"{dir_output}/{fname_out}.png")
                    plt.close(fig)

        # display progress
        hour, min, sec = hour_min_sec(timer() - t_start_p)
        print(f"\n    Subject complete in: {hour} hour, {min} min, and {sec :0.1f} s")

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\nTotal time: {hour} hour, {min} min, and {sec :0.1f} s")


if __name__ == "__main__":
    main()