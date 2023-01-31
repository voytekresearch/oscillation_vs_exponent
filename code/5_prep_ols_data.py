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

    # get data for each parameter and condition (sig chans only)
    dfs = []
    for memory in ['hit', 'miss']:
        df_list = []
        for material in ['words', 'faces']:
            df_list.append(gen_df_alpha(chan_info, material, memory, AP_MODE)[sig_chans])
            df_list.append(gen_df_exp(chan_info, material, memory, AP_MODE, SIG_LEVEL)[sig_chans])
            
        # aggregate data for all params and conditions 
        df_mem = gen_df_ols(df_list[0],df_list[1],df_list[2],df_list[3], memory, sig_chans)

        # add column for memory condition
        df_mem['memory'] = memory

        # add to list of dataframes
        dfs.append(df_mem)

    # join dataframes for hit and miss conditions
    df_ols = pd.concat(dfs, ignore_index=True)
        
    # save dataframe for OLS analysis
    df_ols.to_csv(f"{dir_output}/df_ols.csv")
    
    
def gen_df_ols(df_alpha_w, df_exp_w, df_alpha_f, df_exp_f, memory, sig_chans):
    # add alpha results
    df = df_alpha_w.copy()
    df['diff_alpha_w'] = df["power_diff"]
    df.drop(columns=["power_pre","power_post","power_diff"], inplace=True)
    df['diff_alpha_f'] = df_alpha_f['power_diff']
    
    # add exp results
    df['diff_exp_w'] = df_exp_w['exp_diff']
    df['diff_exp_f'] = df_exp_f['exp_diff']
    
    # add intersection freqeuncy results
    fname_in = f"intersection_results_words_{memory}_{AP_MODE}.npz"
    data_in_word = np.load(f"{PROJECT_PATH}/data/ieeg_intersection_results/{fname_in}", allow_pickle=True)
    fname_in = f"intersection_results_faces_{memory}_{AP_MODE}.npz"
    data_in_face = np.load(f"{PROJECT_PATH}/data/ieeg_intersection_results/{fname_in}", allow_pickle=True)

    df['f_rot_w'] = data_in_word['intersection'][sig_chans]
    df['f_rot_f'] = data_in_face['intersection'][sig_chans]
    
    # add alpha/beta bandpower results
    # laod psd results
    psd_pre_word = np.load(f"{PROJECT_PATH}/data/ieeg_spectral_results/psd_words_hit_prestim.npz")
    psd_post_word = np.load(f"{PROJECT_PATH}/data/ieeg_spectral_results/psd_words_hit_poststim.npz")
    psd_pre_face = np.load(f"{PROJECT_PATH}/data/ieeg_spectral_results/psd_faces_hit_prestim.npz")
    psd_post_face = np.load(f"{PROJECT_PATH}/data/ieeg_spectral_results/psd_faces_hit_poststim.npz")
    freq = psd_post_face['freq']
    diff_psd_word = np.log10(psd_post_word['spectra']) - np.log10(psd_pre_word['spectra'])
    diff_psd_face = np.log10(psd_post_face['spectra']) - np.log10(psd_pre_face['spectra'])
    # trim in alpha band
    _, temp_w = trim_spectrum(freq, diff_psd_word, f_range=bands['alpha'])
    _, temp_f = trim_spectrum(freq, diff_psd_face, f_range=bands['alpha'])
    # average across band
    diff_alpha_bp_word = np.nanmean(temp_w, axis=1)
    diff_alpha_bp_face = np.nanmean(temp_f, axis=1)
    # add to df
    df['diff_alpha_bp_w'] = diff_alpha_bp_word[sig_chans]
    df['diff_alpha_bp_f'] = diff_alpha_bp_face[sig_chans]
    
    # reset index
    df.drop(columns='index', inplace=True)
    df.reset_index(inplace=True)
    
    return df

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

    # correct significance for undected peaks (NaN handling)
    df_alpha.loc[(np.isnan(df_alpha['power_pre']) & np.isnan(df_alpha['power_post'])), 'sig'] = False
    df_alpha.loc[(~np.isnan(df_alpha['power_pre']) & np.isnan(df_alpha['power_post'])), 'sign'] = -1
    df_alpha.loc[(np.isnan(df_alpha['power_pre']) & ~np.isnan(df_alpha['power_post'])), 'sign'] = 1
    
    return df_alpha


if __name__ == "__main__":
    main()

