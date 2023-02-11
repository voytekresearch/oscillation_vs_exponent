"""
Identify channels with significant modulation of alpha/beta bandpower

"""

# Set path
PROJECT_PATH = 'C:/Users/micha/projects/oscillation_vs_exponent/'

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
from stats import gen_random_order, shuffle_rows, comp_resampling_pval
from utils import hour_min_sec, crop_tfr

# dataset details
FS = 512 # meg sampling frequency
TMIN = -1.5 # epoch start time
PATS = ['pat02','pat04','pat05','pat08','pat10','pat11',
         'pat15','pat16','pat17','pat19','pat20','pat21','pat22']

# anlysis parameters
ALPHA_BAND = [8, 20] # alpha/beta frequnecy range
WINDOW = 1 # window size for pre/post stimulus power analysis
EDGE = 0.3 # window size for spectral decomposition
N_ITER = 10000 # random permutation iterations/shuffles
N_DOWNSAMPLE = 128  # number of downsampled tfr time points

def main():
   # time it

    t_start = timer()

    # identify / create directories
    dir_input = f"{PROJECT_PATH}/data/ieeg_tfr/"
    files = os.listdir(dir_input)
    dir_output = f"{PROJECT_PATH}/data/results"
    if not os.path.exists(dir_output): 
        os.makedirs(f"{dir_output}")

    # init dataframe
    columns=['patient','channel','material','memory','p_val','sign']
    df = pd.DataFrame(np.zeros([len(files),len(columns)]), columns=columns)

    # loop through files (channels)
    for i_file, fname_in in enumerate(files):
        # display progress
        t_start_i = timer()
        print(f"\nAnalyzing file {i_file+1}/{len(files)}: \t{time_now()}")
        print(f"    Filename: {fname_in}")

        # determine whether alpha/beta bandpower was task modulation
        p_val, sign = task_modulation(f"{dir_input}/{fname_in}", N_ITER)

        # aggregate results
        f_parts = fname_in.split('_')
        df.loc[i_file] = [f_parts[0], f_parts[3], f_parts[1], f_parts[2], p_val, sign]

        # display progress
        hour, min, sec = hour_min_sec(timer() - t_start_i)
        print(f"    file complete in {hour} hour, {min} min, and {sec :0.1f} s")

    # save results
    df.to_csv(f"{dir_output}/ieeg_modulated_channels.csv")

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\n\n Total Time: \t {hour} hours, {min} minutes, {sec :0.1f} seconds")


def comp_alpha_bandpower(tfr, freq, time, f_range=[8, 20], window=1, edge=0.3, 
                         average=True, print_results=True):
    """Compute alpha bandpower time-series from TFR data.
    
    Parameters
    ----------
    tfr : array 
        Time-frequency representation of the signal. Can be 2 or 3 dimensional.
        If 2d, (freq x time). If 3D (trials x freq x time)
    freq : 1d array
        Frequency values.
    time : 1d array
        Time values. The length of time should match the last dimension of tfr.
    f_range : arrav of 2 floats, optional   
        Alpha frequency range, by default [8, 20].
    window : float, optional
    edge : float, optional
        Window size used power analysis. The tfr will be cropped to remove 
        edge artifacts, by default 0.3.
    average : bool, optional
        If True, average across the alpha band, by default True.
    print_results : bool, optional
        If True, print mean alpha bandpower pre/post-stimulus, by default True.
    
    Returns
    -------
    alpha_pre, alpha_post : arrays
        Alpha bandpower time-series for pre and post-stimulus periods.
    """

    # crop edge artifacts of power analysis
    tfr, time = crop_tfr(tfr, time, [time[0]+(edge/2), time[-1]-(edge/2)])

    # trim tfr in alpha band and average across the band
    _, alpha_band_ts = trim_spectrum(freq, tfr, f_range=f_range)

    # compute mean pre/post-stimulus alpha bandpower
    alpha_pre = alpha_band_ts[..., np.logical_and(time>-window, time<0)]
    alpha_post = alpha_band_ts[..., np.logical_and(time>0, time<window)]

    # average over band (if requested)
    if average:
        alpha_pre = np.mean(alpha_pre, axis=1)
        alpha_post = np.mean(alpha_post, axis=1)

    # print results
    if print_results:
        print(f"mean alpha pre-stimulus: {np.mean(alpha_pre):.2f}")
        print(f"mean alpha post-stimulus: {np.mean(alpha_post):.2f}")
    
    return alpha_pre, alpha_post


def task_modulation(fname_in, n_iter):
    # load tfr
    data_in = np.load(fname_in)

    # skip if only a single trial exists for channel (TEMP - should be handled in step 2)
    if np.ndim(np.squeeze(data_in['tfr'])) < 3: 
        return np.nan, np.nan

    # compute alpha bandpower for pre/post-stim time window
    alpha_pre, alpha_post = comp_alpha_bandpower(np.squeeze(data_in['tfr']), data_in['freq'], 
                                                data_in['time'], f_range=ALPHA_BAND, 
                                                edge=EDGE, window=WINDOW, average=True, 
                                                print_results=False)

    # shuffle conditions
    order = gen_random_order(n_iter, alpha_pre.shape[0]*2)
    alpha_pre_shuf, alpha_post_shuf = shuffle_rows(alpha_pre, alpha_post, order)

    # average shuffled power values over time windows and trials, then compute difference
    alpha_a = np.mean(alpha_post_shuf, axis=(1,2))
    alpha_b = np.mean(alpha_pre_shuf, axis=(1,2))
    alpha_distr = alpha_a - alpha_b

    # compute p value
    alpha_diff = np.mean(alpha_post) - np.mean(alpha_pre)
    p_val, sign = comp_resampling_pval(alpha_distr, alpha_diff)

    return p_val, sign

if __name__ == "__main__":
    main()

