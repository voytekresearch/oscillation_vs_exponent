"""
Identify channels with significant modulation of alpha/beta bandpower

"""

# Set path
PROJECT_PATH = 'C:/Users/micha/projects/oscillation_vs_exponent/'

# ignore mean of empty slice warnings
import warnings
warnings.filterwarnings("ignore")

# Imports - general
import os
import numpy as np
import pandas as pd
from time import time as timer
from time import ctime as time_now
from fooof.utils import trim_spectrum

# Imports - custom
import sys
sys.path.append(f"{PROJECT_PATH}/code")
from stats import run_resampling_analysis
from utils import hour_min_sec

# dataset details
FS = 512 # meg sampling frequency
TMIN = -1.5 # epoch start time
PATIENTS = ['pat02','pat04','pat05','pat08','pat10','pat11',
         'pat15','pat16','pat17','pat19','pat20','pat21','pat22']

# anlysis parameters
ALPHA_BAND = [8, 20] # alpha/beta frequnecy range
N_ITER = 10000 # random permutation iterations/shuffles

def main():
   # time it

    t_start = timer()

    # identify / create directories
    dir_input = f"{PROJECT_PATH}/data/ieeg_psd/"
    files = os.listdir(dir_input)
    dir_output = f"{PROJECT_PATH}/data/results"
    if not os.path.exists(dir_output): 
        os.makedirs(f"{dir_output}")

    # loop through conditions
    dfs = []
    for material in ['words','faces']:
        for memory in ['hit','miss']:
            for patient in PATIENTS:
                # display progress
                t_start_i = timer()
                print(f"\nAnalyzing: {material} - {memory} - {patient}")
                print(f"    Current time: \t{time_now()}")
                
                # init dataframe
                columns=['patient', 'material', 'memory', 'chan_idx', 'p_val', 'sign']
                df = pd.DataFrame(columns=columns)
                
                # load pre- and post-stim psd 
                fname_in = f"{patient}_{material}_{memory}_XXXstim_psd.npz"
                data_pre = np.load(f"{dir_input}/{fname_in.replace('XXX','pre')}")
                data_post = np.load(f"{dir_input}/{fname_in.replace('XXX','post')}")

                # get number of trials and channels
                n_trials = data_pre['psd'].shape[0]
                n_chans = data_pre['psd'].shape[1]
                print(f"    file contains {n_trials} trials and {n_chans} channels...")

                # save metadata
                df['chan_idx'] = np.arange(n_chans)
                df['patient'] = patient
                df['material'] = material
                df['memory'] = memory

                # trim in alpha band
                alpha_pre = np.zeros([n_chans, n_trials])
                alpha_post = np.zeros([n_chans, n_trials])
                # loop through channels
                for i_chan in range(n_chans):
                    # trim
                    _, alpha_band_pre = trim_spectrum(data_pre['freq'], data_pre['psd'][:,i_chan], f_range=ALPHA_BAND)
                    _, alpha_band_post = trim_spectrum(data_post['freq'], data_post['psd'][:,i_chan], f_range=ALPHA_BAND)
                    alpha_pre[i_chan] = np.nanmean(alpha_band_pre, axis=1)
                    alpha_post[i_chan] = np.nanmean(alpha_band_post, axis=1)

                    # determine whether alpha/beta bandpower was task modulation
                    p_val, sign = run_resampling_analysis(alpha_pre[i_chan],
                                                          alpha_post[i_chan], N_ITER)
                    
                    # save results
                    df.loc[i_chan, 'p_val'] = p_val
                    df.loc[i_chan, 'sign'] = sign

                # aggreate results
                dfs.append(df)

                # display progress
                hour, min, sec = hour_min_sec(timer() - t_start_i)
                print(f"    file complete in {hour} hour, {min} min, and {sec :0.1f} s")

    # concatenate results
    df = pd.concat(dfs)

    # save results
    df.to_csv(f"{dir_output}/ieeg_modulated_channels.csv")

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\n\n Total Time: \t {hour} hours, {min} minutes, {sec :0.1f} seconds")


if __name__ == "__main__":
    main()

