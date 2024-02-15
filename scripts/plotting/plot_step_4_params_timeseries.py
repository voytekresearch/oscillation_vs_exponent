"""
This script plots the spectral parameterization results of 
code/4_spectral_parameterization.py. The time-series of each aperiodic
parameter (offset, slope, knee) is plotted for each significant channel.

"""

# Set path
PROJECT_PATH = 'C:/Users/micha/projects/oscillation_vs_exponent/'

# Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time as timer
from fooof import FOOOFGroup

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from utils import hour_min_sec
from plots import plot_ap_params

# Settings
WINDOW = 0.5 # in seconds. Edge effects are removed by cropping the TFR results
T_BASELINE = [-1, 0] # in seconds. These time bins will be averaged for PSD plot
T_ENCODING = [0, 1] # in seconds. These time bins will be averaged for PSD plot

# set plotting parameers
plt.style.use('mpl_styles/default.mplstyle')


def main():
    # time it
    t_start = timer()

    # identify / create directories
    dir_input = f"{PROJECT_PATH}/data/ieeg_tfr_param"
    dir_output = f"{PROJECT_PATH}/figures/params_timeseries"
    if not os.path.exists(dir_output): 
        os.makedirs(dir_output)

    # load task-modulation results
    df_tm = pd.read_csv(f"{PROJECT_PATH}/data/results/ieeg_modulated_channels.csv")
    df = df_tm.loc[(df_tm['material']=='faces') & (df_tm['memory']=='hit')]
    df = df.drop(columns=['material','memory'])
    df = df.loc[df['sig']].reset_index(drop=True)
    
    # load time vector
    fname = f"{PROJECT_PATH}/data/ieeg_tfr/pat02_faces_hit_chan0_tfr.npz"
    data_in = np.load(fname)
    time = data_in['time']

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
                fname = f"{row['patient']}_{material}s_{memory}_chan{row['chan_idx']}_tfr_param_knee"
                
                # display progress
                print(f"\t{material}-{memory}...")
                
                # check if file exists
                if not os.path.exists(f"{dir_input}/{fname}.json"):
                    print(f"\t\t{fname} not found. Skipping...")
                    continue

                # load params
                params = FOOOFGroup()
                params.load(f"{dir_input}/{fname}.json")

                # plot and save
                plot_ap_params(params, time, single_plot=False)
                plt.savefig(f"{dir_output}/{fname}.png")
                plt.close()

        # display progress
        hour, min, sec = hour_min_sec(timer() - t_start_r)
        print(f"\n    Channel complete in: {hour} hour, {min} min, and {sec :0.1f} s")

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\nTotal time: {hour} hour, {min} min, and {sec :0.1f} s")


if __name__ == "__main__":
    main()
