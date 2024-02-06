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
from fooof.bands import Bands

# Imports - custom
from stats import run_resampling_analysis
from utils import hour_min_sec
from tfr_utils import crop_tfr

# dataset details
FS = 512 # meg sampling frequency
TMIN = -1.5 # epoch start time
PATIENTS = ['pat02','pat04','pat05','pat08','pat10','pat11',
         'pat15','pat16','pat17','pat19','pat20','pat21','pat22']

# anlysis parameters
TIME_PRE = [-1.0, 0.0]    # pre-stim
TIME_POST = [0.0, 1.0]    # post-stim
BANDS = Bands({'alpha' : [7, 13]}) # define spectral bands of interest
N_ITER = 10000 # random permutation iterations/shuffles
ALPHA = 0.05 # significance level

def main():
   # time it

    t_start = timer()

    # identify / create directories
    dir_input = f"{PROJECT_PATH}/data/ieeg_tfr/"
    files = os.listdir(dir_input)
    dir_output = f"{PROJECT_PATH}/data/results"
    if not os.path.exists(dir_output): 
        os.makedirs(f"{dir_output}")

    # loop through files
    dfs = []
    for i_file, file in enumerate(files):
        # display progress - every 100 files
        if i_file % 100 == 0:
            hours, minutes, seconds = hour_min_sec(timer() - t_start)
            print(f"\n    Analyzing file {i_file} of {len(files)}...")
            print(f"    Current time: \t{time_now()}")
            print(f"    Elapsed time: \t{hours}h {minutes}m {seconds}s")      
        
        # init dataframe
        columns=['patient', 'material', 'memory', 'chan_idx', 'p_val', 'sign']
        df = pd.DataFrame(columns=columns)
        
        # load pre- and post-stim psd
        data_in = np.load(f"{dir_input}/{file}")
        freq = data_in['freq']
        tfr = data_in['tfr']
        time = data_in['time']

        # save metadata
        f_parts = file.split('_')
        df.loc[0, 'patient'] = f_parts[0]
        df.loc[0, 'chan_idx'] = int(f_parts[3].replace('chan', ''))
        df.loc[0, 'material'] = f_parts[1]
        df.loc[0, 'memory'] = f_parts[2]

        # trim tfr in time windows of interest AND AVERAGE ACROSS TIME
        tfr_pre = np.mean(crop_tfr(tfr, time, TIME_PRE)[0], axis=2)
        tfr_post = np.mean(crop_tfr(tfr, time, TIME_POST)[0], axis=2)

        # trim tf in frequency bands of interest
        alpha_pre = np.mean(trim_spectrum(freq, tfr_pre, BANDS.alpha)[1], axis=1)
        alpha_post = np.mean(trim_spectrum(freq, tfr_post, BANDS.alpha)[1], axis=1)

        # determine whether alpha/beta bandpower was task modulation
        p_val, sign = run_resampling_analysis(alpha_pre, alpha_post, N_ITER)
        
        # save results
        df.loc[0, 'p_val'] = p_val
        df.loc[0, 'sign'] = sign

        # aggreate results
        dfs.append(df)

    # concatenate results
    results = pd.concat(dfs, ignore_index=True)

    # find channels that are task modulated in both material conditions (successful trials)
    results['sig_tm'] = results['p_val'] < ALPHA # determine significance within condition
    sig = results[results['memory']=='hit'].groupby(['patient','chan_idx']).all().reset_index() # find sig chans
    results['sig'] = np.nan # init
    for ii in range(len(sig)):
        results.loc[(results['patient']==sig.loc[ii, 'patient']) & \
                    (results['chan_idx']==sig.loc[ii, 'chan_idx']), 'sig'] \
                        = sig.loc[ii, 'sig_tm'] # add results to df

    # save results
    results.to_csv(f"{dir_output}/ieeg_modulated_channels.csv")

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\n\n Total Time: \t {hour} hours, {min} minutes, {sec :0.1f} seconds")


if __name__ == "__main__":
    main()

