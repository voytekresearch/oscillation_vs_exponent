"""
This script identifies channels with significant task-related modulation of 
total alpha/beta bandpower using permutation testing.

"""

# Imports - standard
import os
import numpy as np
import pandas as pd
from time import time as timer
from time import ctime as time_now
from specparam.utils import trim_spectrum
from specparam.bands import Bands

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from settings import EPOCH_TIMES, BANDS
from stats import run_resampling_analysis
from utils import hour_min_sec
from tfr_utils import crop_tfr

# ignore mean of empty slice warnings
import warnings
warnings.filterwarnings("ignore")

# analysis parameters
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

    # create bands object
    bands = Bands(BANDS)

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
        columns=['patient', 'material', 'memory', 'chan_idx']
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

        # trim tfr in time windows of interest and average across time
        psd_pre = np.mean(crop_tfr(tfr, time, EPOCH_TIMES[1])[0], axis=2)
        psd_post = np.mean(crop_tfr(tfr, time, EPOCH_TIMES[2])[0], axis=2)

        # loop through bands of interst
        for band, f_range in zip(bands.labels, bands.definitions):
            # trim tf in frequency bands of interest and average across freqs
            power_pre = np.mean(trim_spectrum(freq, psd_pre, f_range)[1], axis=1)
            power_post = np.mean(trim_spectrum(freq, psd_post, f_range)[1], axis=1)

            # determine whether bandpower was task modulation
            p_val = run_resampling_analysis(power_pre, power_post, N_ITER)

            # determine sign of effect
            sign = np.sign(np.nanmean(power_post) - np.nanmean(power_pre))
            
            # save results
            df.loc[0, f'pval_{band}'] = p_val
            df.loc[0, f'sign_{band}'] = sign

            # aggreate results
            dfs.append(df)

    # concatenate results
    results = pd.concat(dfs, ignore_index=True)

    # find significant results (p-value < alpha)
    for band in bands.labels:
        results[f'sig_tm_{band}'] = results[f'pval_{band}'] < ALPHA # determine significance within condition

    # find channels that are task modulated in both material conditions 
    results_s = results.loc[results['memory']=='hit'] # select successful trials
    results_s = results_s.groupby(['patient','chan_idx']).all().reset_index() # find channels that are task modulated in both conditions
    results_s.drop(columns=['material', 'memory'], inplace=True)
    for band in bands.labels:
        results_s.rename(columns={f'sig_tm_{band}': f'sig_{band}'}, inplace=True)
        results_s.drop(columns=[f'pval_{band}', f'sign_{band}'], inplace=True)
    results = results.merge(results_s, on=['patient','chan_idx'])
            
    # find channels that are task modulated in all/any frequency bands
    results['sig_all'] = results[[f'sig_{band}' for band in bands.labels]].all(axis=1)
    results['sig_any'] = results[[f'sig_{band}' for band in bands.labels]].any(axis=1)

    # save results
    results.to_csv(f"{dir_output}/ieeg_modulated_channels.csv")

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\n\nTotal Time: \t {hour} hours, {min} minutes, {sec:0.1f} seconds")


if __name__ == "__main__":
    main()

