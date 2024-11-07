"""
The script applies permutation testing to evaluate the significance of the 
spectral parameterization results from 4_spectral_parameterization.py.

"""

# Imports - standard
import os
import numpy as np
import pandas as pd
from timeit import default_timer as timer

from specparam import SpectralGroupModel

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from info import PATIENTS
from settings import (AP_MODE, BANDS, SPEC_PARAM_SETTINGS, N_JOBS, 
                        BAND_POWER_METHOD, LOG_POWER)
from stats import gen_random_order, comp_resampling_pval
from specparam_utils import compute_band_power, compute_adjusted_band_power

# analysis/statistical settings
N_ITER = 1000 # number of iterations for permutation test

def main():
    # display progress
    start_time = timer()
    
    # id directories
    dir_input = f'{PROJECT_PATH}/data/ieeg_psd'
    dir_output = f'{PROJECT_PATH}/data/ieeg_stats/permutation_test'
    if not os.path.exists(dir_output): os.makedirs(dir_output)
    
    # initialize 
    dfs = []

    # load spectral parameterization results
    params = pd.read_csv(f"{PROJECT_PATH}/data/results/spectral_parameters.csv")

    # loop through materials
    for material in ['faces','words']:
        for memory in ['hit']:
            # display progress
            print('---------------------------------------')
            print(f'Condition: {material} - {memory}')
            print('---------------------------------------')

            # loop through patients
            for patient in PATIENTS:
                # TEMP - skip files that have already been processed
                files_completed = os.listdir(dir_output)
                fname_out = f'stats_{patient}_{material}_{memory}.csv'
                if fname_out in files_completed:
                    print(f'Skipping {fname_out} -  already processed')
                    continue

                # display progress
                start_time_p = timer()
                print(f'\nAnalyzing patient: {patient}' )

                # load subject data (pre/post-stim PSDs)
                fname = f'{patient}_{material}_{memory}_xxxstim_psd.npz'
                data_pre = np.load(f"{dir_input}/{fname.replace('xxx','pre')}")
                data_post = np.load(f"{dir_input}/{fname.replace('xxx','post')}")
        
                # get parameters for this patient
                params_i = params.loc[(params['material']==material) & (params['memory']==memory) & (params['patient']==patient)]
                # pivot table on 'epoch' and compute difference for each parameter
                params_p = params_i.pivot_table(index='chan_idx', columns='epoch', values=['exponent','alpha', 'alpha_adj', 'gamma', 'gamma_adj'])

                # run permutation stats
                df_i = resampling_analysis(data_pre['freq'], data_pre['psd'], 
                                            data_post['psd'], params_p)

                # aggregate
                df_i.insert(0, 'patient', patient)
                df_i.insert(2, 'material', material)
                df_i.insert(3, 'memory', memory)
                dfs.append(df_i)
                
                # save results for patient-material-memory
                df_i.to_csv(f"{dir_output}/{fname_out}", index=False)

                # display progress
                print(f'Patient complete. Time: {timer() - start_time_p}')
            
    # aggregate statistical results and save
    df = pd.concat(dfs)
    df.to_csv(f"{dir_output}/stats_all.csv", index=False)

    # display progress
    print('\n---------------------------------------')
    print('Analysis complete!')
    print(f'Total time: {timer() - start_time}')
        
def resampling_analysis(freq, spectra_pre, spectra_post, df_true):
    
    # size up data
    n_trials = spectra_pre.shape[0]
    n_chans = spectra_pre.shape[1]
    
    # initialize
    features = ['exponent'] + [f'{band}{power}' for band in BANDS for power in ['','_adj']]
    pvalues = np.zeros([n_chans, len(features)])

    # loop through channels
    for i_chan in df_true.index: 
        # display progress
        start = timer()
        print(f'    Analyzing channel: {i_chan}/{n_chans}')
        
        # shuffle spectra
        order = gen_random_order(N_ITER, int(n_trials*2))
        spectra_0s, spectra_1s = shuffle_spectra(spectra_pre[:, i_chan], 
                                                 spectra_post[:, i_chan], order)

        # parameterize shuffled spectra
        sgm_0 = run_specparam(freq, spectra_0s)
        sgm_1 = run_specparam(freq, spectra_1s)

        # get spectral features and compute evoked change in each parameter
        features_0 = get_spectral_features(freq, spectra_0s, sgm_0, BANDS)
        features_1 = get_spectral_features(freq, spectra_1s, sgm_1, BANDS)
        df_shuff = features_1 - features_0

        # compute p-values
        for i_feat, feature in enumerate(features):
            true_diff = np.squeeze(np.diff(df_true.loc[i_chan, feature].values))
            pvalues[i_chan, i_feat] = comp_resampling_pval(df_shuff[feature], 
                                                           true_diff)
        # time it
        end = timer()
        print(f"\tcomplete in: {end-start:0.0f} seconds")

    # create dataframe of results
    df = pd.DataFrame({'chan_idx' : np.arange(n_chans)})
    for i_feat, feature in enumerate(features):
        df.insert(i_feat+1, feature, pvalues[:, i_feat])
        
    return df
    
def shuffle_spectra(spectra_0, spectra_1, order):
    # concatenate 2 groups of spectra
    n_spectra = spectra_0.shape[0]
    spectra = np.concatenate([spectra_0, spectra_1])

    # shuffle spectra and average
    spectra_0s = np.zeros([order.shape[0], spectra.shape[1]])
    spectra_1s = spectra_0s.copy()
    for i_iter in range(order.shape[0]):
        spectra_0s[i_iter] = np.nanmedian(spectra[order[i_iter, :n_spectra]], 0)
        spectra_1s[i_iter] = np.nanmedian(spectra[order[i_iter, n_spectra:]], 0)

    return spectra_0s, spectra_1s


def run_specparam(freq, spectra):
    sgm = SpectralGroupModel(**SPEC_PARAM_SETTINGS, aperiodic_mode=AP_MODE, 
                              verbose=False)
    sgm.set_check_data_mode(False)
    sgm.fit(freq, spectra, n_jobs=N_JOBS)
    
    return sgm

    
def get_spectral_features(freq, spectra, sgm, bands):

    # get aperiodic parameters
    df = sgm.to_df(bands)

    # get band power results
    for band in bands:
        # add band power 
        power = compute_band_power(freq, spectra, bands[band], 
                                   method=BAND_POWER_METHOD, 
                                   log_power=LOG_POWER)
        df[band] = power

        # add adjusted band power
        power_adj = compute_adjusted_band_power(sgm, bands[band], 
                                                method=BAND_POWER_METHOD,
                                                log_power=LOG_POWER)
        df[f"{band}_adj"] = power_adj
    
    return df
        

if __name__ == "__main__":
    main()

