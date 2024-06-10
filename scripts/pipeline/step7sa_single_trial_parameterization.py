"""
Sensitivity analysis: SpecParam hyperparameters

Re-run analysis from step7_single_trial_parameterization.py with different
hyperparameter values for SpecParam. This script will run the analysis with
different values for the following hyperparameters: peak_width_limits,
max_n_peaks, and peak_threshold. The results will be saved to a csv file
for each hyperparameter value, and then combined into a single csv file.

"""

# Imports
import os
import numpy as np
import pandas as pd
from specparam import SpectralGroupModel, fit_models_3d

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from settings import N_JOBS, SPEC_PARAM_SETTINGS, FREQ_RANGE, BANDS
from utils import get_start_time, print_time_elapsed
from specparam_utils import (compute_band_power, compute_adjusted_band_power,
                             compute_adj_r2)

# settings
BAND_POWER_METHOD = 'mean'
LOG_POWER = True

def main(specparam_settings, i_run):
    # identify / create directories
    dir_input = f"{PROJECT_PATH}/data/ieeg_psd"
    dir_output = f"{PROJECT_PATH}/data/specparam_sensitivity_analysis"
    if not os.path.exists(dir_output): 
        os.makedirs(f"{dir_output}")
    
    # display progress
    t_start = get_start_time()
    
    # init
    df_list = []

    # loop files
    files = [f for f in os.listdir(dir_input) if not 'epoch' in f]
    for i_file, fname in enumerate(files):
        # display progress
        t_start_c = get_start_time()
        print(f"\nAnalyzing: {fname} ({i_file+1}/{len(files)})")

        # load results for condition
        data_in =  np.load(f"{dir_input}/{fname}")
        spectra = data_in['psd']
        freq = data_in['freq']
        
        # parameterize (fit both with and without knee parametere)
        for ap_mode in ['fixed', 'knee']:
            # apply SpecParam
            sgm = SpectralGroupModel(**specparam_settings, 
                                     aperiodic_mode=ap_mode, verbose=False)
            sgm.set_check_modes(check_data=False)
            params = fit_models_3d(sgm, freq, spectra, 
                                   n_jobs=N_JOBS, freq_range=FREQ_RANGE)

            # convert results to dataframe and store
            df = pd.concat([sm.to_df(0) for sm in params])
            f_parts = fname.split('_')
            df.insert(0, 'patient', f_parts[0])
            df.insert(1, 'trial', np.repeat(np.arange(spectra.shape[0]), spectra.shape[1]))
            df.insert(2, 'chan_idx', np.tile(np.arange(spectra.shape[1]), spectra.shape[0]))
            df.insert(3, 'material', f_parts[1])
            df.insert(4, 'memory', f_parts[2])
            df.insert(5, 'epoch', f_parts[3].replace('stim', ''))
            df.insert(6, 'ap_mode', ap_mode)
            df.insert(7, 'peak_width_limits', specparam_settings['peak_width_limits'][1])
            df.insert(8, 'max_n_peaks', specparam_settings['max_n_peaks'])
            df.insert(9, 'peak_threshold', specparam_settings['peak_threshold'])

            # add adjusted r-squared
            r2_adj = []
            for sgm in params:
                for i_model in range(len(sgm)):
                    sm = sgm.get_model(i_model, regenerate=True)
                    if np.isnan(sm.r_squared_):
                        r2_adj.append(np.nan)
                    else:
                        r2_adj.append(compute_adj_r2(sm))
            df['r2_adj'] = r2_adj

            # compute power and add to dataframe
            for band in BANDS:
                power_total = []
                power_adjusted = []
                for ii, sm in enumerate(params):
                    # compute total band power 
                    power = compute_band_power(freq, spectra[ii],
                                                BANDS[band], method=BAND_POWER_METHOD,
                                                log_power=LOG_POWER)
                    power_total.append(power)

                    # compute adjusted band power
                    power = compute_adjusted_band_power(sm, BANDS[band], 
                                                        method=BAND_POWER_METHOD,
                                                        log_power=LOG_POWER)
                    power_adjusted.append(power)
                
                # add to dataframe
                df[f"{band}"] = np.concatenate(power_total)
                df[f"{band}_adj"] = np.concatenate(power_adjusted)

            # store
            df_list.append(df)

        # display progress
        print_time_elapsed(t_start_c)

    # combine results and save
    results = pd.concat(df_list)
    results.to_csv(f"{dir_output}/psd_trial_params_{i_run}.csv", index=False)

    # display progress
    print("\n\nAnalysis complete!")
    print_time_elapsed(t_start)
 
        
if __name__ == "__main__":
    """
    Sensitivity analysis for SpecParam hyperparameters.
    """

    # hyperparameter settings
    hyperparameters = ['peak_width_limits', 'max_n_peaks', 'peak_threshold']
    peak_width_limits = [[2, 4], [2, 8], [2, 12], [2, 16], [2, 20]]
    max_n_peaks = [0, 2, 4, 6, 8]
    peak_threshold = [1, 2, 3, 4, 5]
    
    # run main with each hyperparameter value
    i_run = 0
    n_runs = len(peak_width_limits) + len(max_n_peaks) + len(peak_threshold)
    for hyperparameter in hyperparameters:
        for value in locals()[hyperparameter]:
            print(f"\n\nSensitivity analysis: run {i_run+1}/{n_runs}") # display progress
            specparam_settings = SPEC_PARAM_SETTINGS.copy() # copy settings
            specparam_settings[hyperparameter] = value # update settings
            main(specparam_settings, i_run) # run main
            i_run += 1

    # combine results
    files = os.listdir(f"{PROJECT_PATH}/data/specparam_sensitivity_analysis")
    df_list = []
    for f in files:
        df_list.append(pd.read_csv(f"{PROJECT_PATH}/data/specparam_sensitivity_analysis/{f}"))
    results = pd.concat(df_list)
    results.to_csv(f"{PROJECT_PATH}/data/results/psd_trial_params_sa.csv", index=False)

    # # run each combination of hyperparameters - long runtime
    # i_run = 0
    # n_runs = len(peak_width_limits) * len(max_n_peaks) * len(peak_threshold)
    # for peak_width in peak_width_limits:
    #     for max_n_peaks in max_n_peaks:
    #         for peak_threshold in peak_threshold:
    #             print(f"\n\nSensitivity analysis: run {i_run+1}/{n_runs}")

    #             # update settings
    #             specparam_settings = SPEC_PARAM_SETTINGS.copy()
    #             for hyperparameter in hyperparameters:
    #                 specparam_settings[hyperparameter] = locals()[hyperparameter]
                
    #             # run main
    #             main(specparam_settings, i_run)
    #             i_run += 1
