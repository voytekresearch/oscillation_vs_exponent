"""
This script parameterizes the spectral results from 2_time_frequency_analysis.py

"""

# Imports
import os
import numpy as np
import pandas as pd
from fooof import FOOOFGroup
from time import time as timer

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from info import N_JOBS, FREQ_RANGE, SPEC_PARAM_SETTINGS
from utils import hour_min_sec

# Settings
RUN_TFR = False # run TFR parameterization (takes a long time)

# SpecParam hyperparameters
AP_MODE = ['knee'] # ['fixed', 'knee'] # aperiodic mode
DECOMP_METHOD = 'tfr' # paraneterize PSDs or average TFRs

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

def param_group_psd_results():
    # identify / create directories
    dir_input = f"{PROJECT_PATH}/data/ieeg_spectral_results"
    dir_output = f"{PROJECT_PATH}/data/ieeg_psd_param"
    if not os.path.exists(dir_output): 
        os.makedirs(f"{dir_output}/fooof_reports")
    
    # display progress
    t_start = timer()
    
    # loop through conditions
    files = [f for f in os.listdir(dir_input) if f.startswith(DECOMP_METHOD)]
    for fname in files:
        # display progress
        t_start_c = timer()
        print(f"\tAnalyzing: {fname}")

        # load results for condition
        data_in =  np.load(f"{dir_input}/{fname}")
        spectra = data_in['spectra']
        freq = data_in['freq']
        
        # parameterize (fit both with and without knee parametere)
        for ap_mode in AP_MODE:
            fg = FOOOFGroup(**SPEC_PARAM_SETTINGS, aperiodic_mode=ap_mode, verbose=False)
            fg.set_check_data_mode(False)
            fg.fit(freq, spectra, n_jobs=N_JOBS, freq_range=FREQ_RANGE)
            
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
            fg.fit(freq, tfr.T, n_jobs=N_JOBS, freq_range=FREQ_RANGE)
            
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