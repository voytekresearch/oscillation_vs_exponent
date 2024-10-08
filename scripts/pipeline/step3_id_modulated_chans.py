"""
This script identifies channels with significant task-related modulation of 
total canonical oscillatory bandpower using permutation testing.

"""

# Imports - standard
import os
import numpy as np
import pandas as pd
from time import time as timer
from time import ctime as time_now
from specparam.utils import trim_spectrum
from specparam.bands import Bands
from scipy.stats import permutation_test

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from settings import BANDS
from utils import hour_min_sec
from stats import mean_difference

# ignore mean of empty slice warnings
import warnings
warnings.filterwarnings("ignore")

# analysis parameters
N_ITER = 1000 # random permutation iterations/shuffles
ALPHA = 0.05 # significance level

def main():
   # time it
    t_start = timer()

    # identify / create directories
    dir_input = f"{PROJECT_PATH}/data/ieeg_psd/"
    files = os.listdir(dir_input)
    dir_output = f"{PROJECT_PATH}/data/results"
    if not os.path.exists(dir_output): 
        os.makedirs(f"{dir_output}")

    # create bands object
    bands = Bands(BANDS)
        
    # init dataframe
    results = pd.DataFrame()

    # loop through files
    fnames = [f for f in files if "prestim" in f]
    for i_file, file in enumerate(fnames):
        # display progress - every 100 files
        hours, minutes, seconds = hour_min_sec(timer() - t_start)
        print(f"\nAnalyzing file {i_file} of {len(fnames)}...")
        print(f"    Current time: \t{time_now()}")
        print(f"    Elapsed time: \t{hours}h {minutes}m {seconds}s")      
        
        # init dataframe
        columns=['patient', 'material', 'memory', 'chan_idx']
        df = pd.DataFrame(columns=columns)
        
        # load pre- and post-stim psd
        data_pre = np.load(f"{dir_input}/{file}")
        data_post = np.load(f"{dir_input}/{file.replace('pre', 'post')}")
        psd_pre = data_pre['psd']
        psd_post = data_post['psd']
        freq = data_pre['freq']

        # loop through channels
        for i_chan in range(psd_pre.shape[1]):
            # save metadata
            f_parts = file.split('_')
            df.loc[i_chan, 'patient'] = f_parts[0]
            df.loc[i_chan, 'material'] = f_parts[1]
            df.loc[i_chan, 'memory'] = f_parts[2]
            df.loc[i_chan, 'chan_idx'] = i_chan

            # loop through bands of interst
            for band, f_range in zip(bands.labels, bands.definitions):
                # trim psd in frequency bands of interest and average across freqs
                power_pre = np.mean(trim_spectrum(freq, psd_pre[:, i_chan], 
                                                 f_range)[1], axis=1)
                power_post = np.mean(trim_spectrum(freq, psd_post[:, i_chan],
                                                  f_range)[1], axis=1)

                # correct for nan
                if (np.isnan(power_pre).all()) | (np.isnan(power_post).all()):
                    # save results
                    df.loc[i_chan, f'{band}_pre'] = np.nan
                    df.loc[i_chan, f'{band}_post'] = np.nan
                    df.loc[i_chan, f'{band}_pval'] = np.nan
                    df.loc[i_chan, f'{band}_sign'] = np.nan
                    continue

                # determine whether bandpower was task modulation
                stats = permutation_test([power_pre, power_post], 
                                         statistic=mean_difference, 
                                         permutation_type='samples',
                                         n_resamples=N_ITER,
                                         alternative='two-sided',
                                         random_state=0)

                # determine sign of effect
                sign = np.sign(np.nanmean(power_post) - np.nanmean(power_pre))

                # save results
                df.loc[i_chan, f'{band}_pre'] = np.nanmean(power_pre)
                df.loc[i_chan, f'{band}_post'] = np.nanmean(power_post)
                df.loc[i_chan, f'{band}_pval'] = stats.pvalue
                df.loc[i_chan, f'{band}_sign'] = sign

        # aggreate results
        results = pd.concat([results, df], ignore_index=True)

    # find significant results (p-value < alpha)
    for band in bands.labels:
        results[f'{band}_sig'] = results[f'{band}_pval'] < ALPHA # determine significance within condition

    # # save intermediate results
    results.to_csv(f"{dir_output}/band_power_statistics.csv")
    # results = pd.read_csv(f"{dir_output}/band_power_statistics.csv", index_col=0)

    # find channels that are task modulated in both material conditions 
    results_s = results.loc[results['memory']=='hit'] # select successful trials
    results_s = results_s.groupby(['patient','chan_idx']).all().reset_index() # find channels that are task modulated in both conditions
    results_s.drop(columns=['material', 'memory'], inplace=True) # drop unnecessary columns
    for band in bands.labels: # drop unnecessary columns
        results_s.rename(columns={f'{band}_sig' : f'sig_{band}'}, inplace=True)
        results_s.drop(columns=[f'{band}_pre', f'{band}_post', f'{band}_pval', 
                                f'{band}_sign'], inplace=True)

    # find channels that are task modulated in all/any frequency bands
    results_s['sig_all'] = results_s[[f'sig_{band}' for band in bands.labels]].all(axis=1)
    results_s['sig_any'] = results_s[[f'sig_{band}' for band in bands.labels]].any(axis=1)

    # save results
    results_s.to_csv(f"{dir_output}/ieeg_modulated_channels.csv")

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\n\nTotal Time: \t {hour} hours, {min} minutes, {sec:0.1f} seconds")


if __name__ == "__main__":
    main()

