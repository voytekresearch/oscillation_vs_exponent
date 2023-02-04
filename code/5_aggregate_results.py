# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 11:46:53 2021

@author: micha
"""

# ===== Imports =====
import os
import numpy as np
import pandas as pd

from fooof import FOOOFGroup
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fg
from fooof.utils import trim_spectrum


# ===== parameters =====
# = project params =

# paths and filenames
PROJECT_PATH = 'C:/Users/micha/projects/oscillation_vs_exponent/'
DIR_FIG = f"{PROJECT_PATH}/figures/manuscript_figures"
DIR_STATS = f"{PROJECT_PATH}/data/ieeg_stats/permutation_test"
FNAME_STATS = f"{PROJECT_PATH}/data/fellner_stats/fellner_stats_df.pkl"

# dataset details
FS = 512 # sampling frequency
TMIN = -1.5 # epoch start time
PATIENTS = ['pat02','pat04','pat05','pat08','pat10','pat11','pat15','pat16',
            'pat17','pat19','pat20','pat21','pat22'] # subject IDs

# = spectral analysis parameters =
# define spectral bands of interest
bands = Bands({'alpha' : [8, 20]})

# define aperiodic mode
AP_MODE = 'knee'

# significant level for permutation testing
SIG_LEVEL = 0.01

def main():
    # id directories
    dir_output = f"{PROJECT_PATH}/data/results"
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    # Identify channels with reported effects (Fellner, 2019) 
    df_stats = np.load(FNAME_STATS, allow_pickle=True)
    sig_chans = df_stats['h_cond_alphabeta'] | df_stats['h_cond_gamma']

    # load channel info
    chan_info = pd.read_pickle(F"{PROJECT_PATH}/data/ieeg_metadata/ieeg_channel_info.pkl")
    chan_info.drop(columns=['index'], inplace=True)
    chan_info['unique_id'] = chan_info['patient'] + '_' + chan_info['chan_idx'].astype(str)

    # get data for each parameter and condition
    df_list = []
    for memory in ['hit', 'miss']:
        for material in ['words', 'faces']:
            # get exponent and alpha results
            df_alpha = gen_df_alpha(chan_info, material, memory, AP_MODE)
            df = gen_df_exp(chan_info, material, memory, AP_MODE, SIG_LEVEL)
            
            # join exponent and alpha results
            df['alpha_pre'] = df_alpha['power_pre']
            df['alpha_post'] = df_alpha['power_post']
            df['alpha_diff'] = df_alpha['power_diff']
            df['peak_present'] = df_alpha['peak_present']

            # add intersection frequnecy results
            fname_in = f"intersection_results_{material}_{memory}_{AP_MODE}.npz"
            data_in = np.load(f"{PROJECT_PATH}/data/ieeg_intersection_results/{fname_in}", allow_pickle=True)
            df['f_rot'] = data_in['intersection']

            # add alpha/beta bandpower results
            df['alpha_bp_diff'] = comp_bandpower_change(material, memory)

            # add condition info (material, memory, ap_mode)
            df['material'] = material
            df['memory'] = memory
            df['ap_mode'] = AP_MODE
        
            # add to list
            df_list.append(df)

    # join dataframes for hit and miss conditions
    df_ols = pd.concat(df_list, ignore_index=True)
        
    # save dataframe for OLS analysis
    df_ols.to_csv(f"{dir_output}/df_ols_allchans.csv")

    # save dataframe for OLS analysis (significant channels only)
    df_ols_sig = df_ols[np.tile(sig_chans, 4)]
    df_ols_sig.to_csv(f"{dir_output}/df_ols.csv")


def gen_df_exp(chan_info, material, memory, ap_mode, sig_level):
    # initialize dataframe
    df = chan_info.copy()

    # add exponent results for each epoch
    for epoch in ['prestim', 'poststim']:
        # load spectral results
        param = FOOOFGroup()
        fname_in = f"{material}_{memory}_{epoch}_params_{ap_mode}.json"
        param.load(f"{PROJECT_PATH}/data/ieeg_psd_param/{fname_in}")
    
        # aggregate in dataframe with chan info
        df[f"exp_{epoch.replace('stim','')}"] = param.get_params('aperiodic','exponent')

    # compute change in exponent
    df['exp_diff'] = df['exp_post'] - df['exp_pre']
    
    return df

def gen_df_alpha(chan_info, material, memory, ap_mode):
    # initialize dataframe
    df_alpha = chan_info.copy()

    # add alpha peak results for each epoch
    for epoch in ['prestim', 'poststim']:
        # load spectral results
        param = FOOOFGroup()
        fname_in = f"{material}_{memory}_{epoch}_params_{ap_mode}.json"
        param.load(f"{PROJECT_PATH}/data/ieeg_psd_param/{fname_in}")
        
        # get alpha peak info and add to dataframe
        alpha = get_band_peak_fg(param, [8, 20])
        df_alpha[f"power_{epoch.replace('stim','')}"] = alpha[:,1]
        
    # compute change in alpha
    df_alpha['power_diff'] = df_alpha['power_post'] - df_alpha['power_pre']
    
    # label (un)detected alpha peaks (NaN handling)
    df_alpha['peak_present'] = np.nan
    df_alpha.loc[(np.isnan(df_alpha['power_pre']) & np.isnan(df_alpha['power_post'])), 'peak_present'] = 0
    df_alpha.loc[(~np.isnan(df_alpha['power_pre']) & np.isnan(df_alpha['power_post'])), 'peak_present'] = 1
    df_alpha.loc[(np.isnan(df_alpha['power_pre']) & ~np.isnan(df_alpha['power_post'])), 'peak_present'] = 2
    df_alpha.loc[(~np.isnan(df_alpha['power_pre']) & ~np.isnan(df_alpha['power_post'])), 'peak_present'] = 3

    return df_alpha

def comp_bandpower_change(material, memory):
    # laod psd results
    psd_pre = np.load(f"{PROJECT_PATH}/data/ieeg_spectral_results/psd_{material}_{memory}_prestim.npz")
    psd_post = np.load(f"{PROJECT_PATH}/data/ieeg_spectral_results/psd_{material}_{memory}_poststim.npz")
    
    # trim in alpha band
    _, alpha_bp_pre = trim_spectrum(psd_pre['freq'], psd_pre['spectra'], f_range=bands['alpha'])
    _, alpha_bp_post = trim_spectrum(psd_post['freq'], psd_post['spectra'], f_range=bands['alpha'])

    # compute change in average power across band
    diff_alpha_bp = np.nanmean(alpha_bp_post, axis=1) - np.nanmean(alpha_bp_pre, axis=1)
    
    return diff_alpha_bp


if __name__ == "__main__":
    main()