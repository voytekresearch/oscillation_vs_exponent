"""
This script parameterizes the spectral results from 
2_time_frequency_analysis.py. It differs from the original script in that it
parameterizes the single-trial PSDs rather than the average PSDs. 


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

def main():
    # identify / create directories
    dir_input = f"{PROJECT_PATH}/data/ieeg_psd"
    dir_output = f"{PROJECT_PATH}/data/results"
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
            sgm = SpectralGroupModel(**SPEC_PARAM_SETTINGS, 
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
                    power = compute_adjusted_band_power(freq, spectra[ii],
                                                        sm, BANDS[band], 
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
    results.to_csv(f"{dir_output}/psd_trial_params.csv", index=False)

    # display progress
    print("\n\nAnalysis complete!")
    print_time_elapsed(t_start)
 
        
if __name__ == "__main__":
    main()