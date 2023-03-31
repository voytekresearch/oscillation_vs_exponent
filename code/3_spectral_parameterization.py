# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 17:35:02 2021

@author: micha

Data Repo: https://osf.io/3csku/
Associated Paper: https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000403

This script parameterizes the spectral results from 
ieeg_2_time_frequency_analysis.py

"""

# Set path
PROJECT_PATH = 'C:/Users/micha/projects/oscillation_vs_exponent/'

# Imports
import os
import numpy as np
import pandas as pd
from fooof import FOOOFGroup
from fooof.utils.data import interpolate_spectrum
from time import time as timer

# Imports - custom
from utils import hour_min_sec

# Settings
LINE_NOISE_RANGE = [45,55] # freq range to interpolate
RUN_TFR = True # run TFR parameterization
N_SAMPLES = 2**7 # number of time samples after downsampling

# SpecParam hyperparameters
N_JOBS = -1 # number of jobs for parallel processing
SPEC_PARAM_SETTINGS = {
    'peak_width_limits' :   [2, np.inf], # default : (0.5, 12.0) - recommends at least frequency resolution * 2
    'min_peak_height'   :   0, # default : 0
    'max_n_peaks'       :   4, # default : inf
    'peak_threshold'    :   4} # default : 2.0
AP_MODE = ['knee'] # ['fixed', 'knee'] # aperiodic mode

# FOOOF is causing some warnings about ragged arrays
import warnings
warnings.filterwarnings("ignore")

def main():
    
    # parameterize PSDs
    print('\nParameterizing PSDs...')
    param_group_psd_results()

    # parameterize TFRs
    if RUN_TFR:
        print('\nParameterizing TFRs...')
        parameterize_tfr()
        # param_group_tfr_results() # computationally intensive

def param_group_psd_results():
    # identify / create directories
    dir_input = f"{PROJECT_PATH}/data/ieeg_spectral_results"
    dir_output = f"{PROJECT_PATH}/data/ieeg_psd_param"
    if not os.path.exists(dir_output): 
        os.makedirs(f"{dir_output}/fooof_reports")
    
    # display progress
    t_start = timer()
    
    # loop through conditions
    files = [f for f in os.listdir(dir_input) if f.startswith('psd_')]
    for fname in files:
        # display progress
        t_start_c = timer()
        print(f"\tAnalyzing: \t{fname}")

        # load results for condition
        data_in =  np.load(f"{dir_input}/{fname}")
        spectra = data_in['spectra']
        freq = data_in['freq']
        
        # interpolate psd for frequency range that includes line noise
        spectra = np.zeros_like(spectra_raw)
        for chan in range(len(spectra)):
            _, spectra[chan] = interpolate_spectrum(freq, spectra_raw[chan], 
                                                    LINE_NOISE_RANGE)
        
        # parameterize (fit both with and without knee parametere)
        for ap_mode in AP_MODE:
            print(f"\t\tParameterizing with '{ap_mode}' aperiodic mode...")
            fg = FOOOFGroup(**SPEC_PARAM_SETTINGS, aperiodic_mode=ap_mode, verbose=False)
            fg.set_check_data_mode(False)
            fg.fit(freq, spectra, n_jobs=N_JOBS)
            
            # save results 
            fname_out = fname.replace('.npz', f'_params_{ap_mode}')
            fg.save(f"{dir_output}/{fname_out}", save_results=True, 
                    save_settings=True, save_data=True)
            fg.save_report(f"{dir_output}/fooof_reports/{fname_out}")

        # display progress
        hour, min, sec = hour_min_sec(timer() - t_start_c)
        print(f"\t\tCondition completed in {hour} hour, {min} min, and {sec :0.1f} s")

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"Total PSD analysis time: {hour} hour, {min} min, and {sec :0.1f} s")

def param_group_tfr_results():
    # time it
    t_start = timer()

    # identify / create directories
    dir_input = f"{PROJECT_PATH}/data/ieeg_spectral_results"
    dir_output = f"{PROJECT_PATH}/data/ieeg_tfr_param"
    if not os.path.exists(dir_output): 
        os.makedirs(f"{dir_output}/fooof_reports")
        
    # loop through conditions
    conditions = ['words_hit', 'faces_hit','words_miss', 'faces_miss']
    for cond in conditions:
        # display progress
        print(f"\tAnalyzing condition: \t{cond.replace('_', ', ')}...")
        t_start_c = timer()
    
        # load group TFR results
        data_in = np.load(f"{dir_input}/tfr_{cond}.npz")
        freq = data_in['freq']
        tfr_in = data_in['tfr']
        
        # reshape TFR for parameterization
        tfr = tfr_in.reshape([tfr_in.shape[0]*tfr_in.shape[1],tfr_in.shape[2]])
        
        # parameterize (fit both with and without knee parametere)
        for ap_mode in ['knee']: # ['fixed', 'knee']:
            # print(f"\t\tParameterizing with '{ap_mode}' aperiodic mode...")
            fg = FOOOFGroup(**SPEC_PARAM_SETTINGS, aperiodic_mode=ap_mode, verbose=False)
            fg.set_check_data_mode(False)
            fg.fit(freq, tfr, n_jobs=N_JOBS)
            
            # save results and report
            fname_out = 'tfr_%s_params_%s' %(cond, ap_mode)
            fg.save(f"{dir_output}/{fname_out}", save_results=True, 
                    save_settings=True, save_data=True)
            fg.save_report(f"{dir_output}/fooof_reports/{fname_out}")

        # display progress
        hour, min, sec = hour_min_sec(timer() - t_start_c)
        print(f"\t\tCondition ompleted in {hour} hour, {min} min, and {sec :0.1f} s")

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"Total TFR analysis time: {hour} hour, {min} min, and {sec :0.1f} s")
 

def parameterize_tfr():
    # time it
    t_start = timer()

    # identify / create directories
    dir_input = f"{PROJECT_PATH}/data/ieeg_tfr"
    dir_output = f"{PROJECT_PATH}/data/ieeg_tfr_param"
    if not os.path.exists(f"{dir_output}/fooof_reports"): 
        os.makedirs(f"{dir_output}/fooof_reports")

    # load alpha/beta bandpower modulation results (resampling ananlysis)
    results = pd.read_csv(f"{PROJECT_PATH}/data/results/ieeg_modulated_channels.csv")
    df = results[results['sig']==1].reset_index(drop=True)
    
    # loop through significant channels
    for i_chan in range(len(df)):
        # Check for TFR results
        fname = f"{df.loc[i_chan, 'patient']}_{df.loc[i_chan, 'material']}_" + \
                    f"{df.loc[i_chan, 'memory']}_chan{df.loc[i_chan, 'chan_idx']}_tfr.npz"

        # display progress
        print(f"    Analyzing file {i_chan}/{len(df)}") 
        print(f"\t{fname}")
        t_start_c = timer()
        
        # load tfr
        data_in = np.load(f"{dir_input}/{fname}")
        tfr_in = data_in['tfr']
        freq = data_in['freq']
        
        # average over trials
        tfr = np.squeeze(np.mean(tfr_in, axis=0))
        
        # parameterize
        for ap_mode in AP_MODE:
            # print(f"\t\tParameterizing with '{ap_mode}' aperiodic mode...")
            fg = FOOOFGroup(**SPEC_PARAM_SETTINGS, aperiodic_mode=ap_mode, verbose=False)
            fg.set_check_data_mode(False)
            fg.fit(freq, tfr.T, n_jobs=N_JOBS)
            
            # save results and report
            fname_out = fname.replace('.npz','_param_%s' %ap_mode)
            fg.save(f"{dir_output}/{fname_out}", save_results=True, 
                    save_settings=True)
            fg.save_report(f"{dir_output}/fooof_reports/{fname_out}")

        # display progress
        hour, min, sec = hour_min_sec(timer() - t_start_c)
        print(f"\tFile completed in {hour} hour, {min} min, and {sec :0.1f} s")

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"Total TFR analysis time: {hour} hour, {min} min, and {sec :0.1f} s")
     
        
if __name__ == "__main__":
    main()