"""
Sensitivity analysis for SpecParam hyperparameters.

"""

# Imports
import os
import numpy as np
import pandas as pd
from specparam import SpectralGroupModel

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from settings import N_JOBS, SPEC_PARAM_SETTINGS, FREQ_RANGE, BANDS
from utils import get_start_time, print_time_elapsed
from specparam_utils import (compute_band_power, compute_adjusted_band_power, 
                             compute_adj_r2)

# analysis settings - compute band power
BAND_POWER_METHOD = 'mean'
LOG_POWER = True


def main():
    # display progress
    t_start = get_start_time()

    # identify / create directories
    dir_output = f"{PROJECT_PATH}/data/specparam_sensitivity_analysis"
    dir_results = f"{PROJECT_PATH}/data/results"
    for path in [dir_output, dir_results]:
        if not os.path.exists(path): 
            os.makedirs(path)

    # hyperparameter settings
    hyperparameters = ['peak_width_limits', 'max_n_peaks', 'peak_threshold']
    peak_width_limits = [[2, 4], [2, 8], [2, 12], [2, 16], [2, 20]]
    max_n_peaks = [0, 2, 4, 6, 8]
    peak_threshold = [1, 2, 3, 4, 5]
    
    # run step 4 with each hyperparameter value
    i_run = 0
    n_runs = len(peak_width_limits) + len(max_n_peaks) + len(peak_threshold)
    for hyperparameter in hyperparameters:
        for value in locals()[hyperparameter]:
            print(f"\n\nSensitivity analysis, run {i_run+1}/{n_runs}") # display progress
            specparam_settings = SPEC_PARAM_SETTINGS.copy() # copy settings
            specparam_settings[hyperparameter] = value # update settings
            step_4(specparam_settings, i_run, dir_output) # run main
            i_run += 1

    # combine results across runs
    df_list = []
    for f in os.listdir(dir_output):
        df_list.append(pd.read_csv(f"{dir_output}/{f}"))
    results = pd.concat(df_list)
    results.to_csv(f"{dir_results}/spectral_parameters_sa.csv", index=False)

    # display progress
    print("\n\nAnalysis complete!")
    print_time_elapsed(t_start, "Total analysis time: ")


def step_4(specparam_settings, i_run, dir_output):
    # display progress
    t_start_c = get_start_time()

    # identify / create directories
    dir_input = f"{PROJECT_PATH}/data/ieeg_spectral_results"
    
    # loop through conditions
    df_list = []
    files = [f for f in os.listdir(dir_input) if f.startswith('psd') & (not 'epoch' in f)]
    for i_file, fname in enumerate(files):
        # display progress
        print(f"\tAnalyzing file {i_file+1}/{len(files)}: {fname}")

        # load results for condition
        data_in =  np.load(f"{dir_input}/{fname}")
        spectra = data_in['spectra']
        freq = data_in['freq']

        # load stats - analyze only task-modulated channels
        df_stats = load_stats()
        spectra = spectra[df_stats['sig_both']]
        df_info = df_stats.loc[df_stats['sig_both'], ['patient', 'chan_idx']].reset_index(drop=True)
        
        # parameterize (fit both with and without knee parametere)
        for ap_mode in ['fixed', 'knee']:
            # fit model
            sgm = SpectralGroupModel(**specparam_settings, 
                                     aperiodic_mode=ap_mode, verbose=False)
            sgm.set_check_modes(check_freqs=False, check_data=False)
            sgm.fit(freq, spectra, n_jobs=N_JOBS, freq_range=FREQ_RANGE)

            # convert results to dataframe and store
            df_params = sgm.to_df(0)
            df = pd.concat([df_info, df_params], axis=1)
            f_parts = fname.split('_')
            df.insert(2, 'material', f_parts[1])
            df.insert(3, 'memory', f_parts[2])
            df.insert(4, 'epoch', f_parts[3].replace('stim.npz', ''))
            df.insert(5, 'ap_mode', ap_mode)
            df.insert(6, 'peak_width_limits', specparam_settings['peak_width_limits'][1])
            df.insert(7, 'max_n_peaks', specparam_settings['max_n_peaks'])
            df.insert(8, 'peak_threshold', specparam_settings['peak_threshold'])

            # add adjusted r-squared
            r2_adj = []
            for i_model in range(len(sgm)):
                sm = sgm.get_model(i_model, regenerate=True)
                if np.isnan(sm.r_squared_):
                    r2_adj.append(np.nan)
                else:
                    r2_adj.append(compute_adj_r2(sm))
            df.insert(len(df.columns), 'r2_adj', r2_adj)

            # compute toatl and aperiodic-adjusted power and add to dataframe
            for band in BANDS:
                total_power = compute_band_power(freq, spectra,
                                            BANDS[band], log_power=LOG_POWER,
                                            method=BAND_POWER_METHOD)
                adjusted_power = compute_adjusted_band_power(sm, BANDS[band], 
                                                    method=BAND_POWER_METHOD,
                                                    log_power=LOG_POWER)
                df.insert(len(df.columns), f'{band}_total_power', total_power)
                df.insert(len(df.columns), f'{band}_adjusted_power', 
                          adjusted_power)
            # store
            df_list.append(df)

    # combine results and save
    results = pd.concat(df_list)
    results.to_csv(f"{dir_output}/spectral_parameters_{i_run}.csv", index=False)

    # display progress
    print_time_elapsed(t_start_c, "\n\tRun completed in: ")
     

def load_stats():
    # load stats
    fname = f"{PROJECT_PATH}/data/results/band_power_statistics.csv"
    df_stats = pd.read_csv(fname, index_col=0)
    df_stats = df_stats.loc[df_stats['memory']=='hit']

    # compute joint significance within material
    df_stats['sig_all'] = df_stats['alpha_sig'] & df_stats['gamma_sig'] # both bands within material
    df_stats['sig_any'] = df_stats['alpha_sig'] | df_stats['gamma_sig'] # either band within material

    # pivot table to compute joint significance across materials
    values = ['alpha_sig', 'gamma_sig', 'sig_any', 'sig_all']
    df_stats = df_stats.pivot_table(index=['patient', 'chan_idx'], 
                                    columns='material', values=values)
    df_stats.columns = [f"{col[0]}_{col[1]}" for col in df_stats.columns]
    df_stats.reset_index(inplace=True)
    for col in df_stats.columns[2:]:
        df_stats[col] = df_stats[col].astype(bool) # reset to booleen

    # compute joint significance across materials
    df_stats['sig_any'] = df_stats['sig_any_faces'] | \
        df_stats['sig_any_words'] # either band, either material
    df_stats['sig_both'] = df_stats['sig_all_faces'] | \
        df_stats['sig_all_words'] # both band, either material
    df_stats['sig_all'] = df_stats['sig_all_faces'] & \
        df_stats['sig_all_words'] # both band, both material
    
    return df_stats


if __name__ == "__main__":
    main()
    