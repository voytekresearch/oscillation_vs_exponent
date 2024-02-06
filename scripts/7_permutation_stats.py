"""
The script applies permutation testing to evaluate the significance of the 
spectral parameterization results from 4_spectral_parameterization.py.

"""

# Imports - standard
import os
import numpy as np
import pandas as pd
from timeit import default_timer as timer
from fooof import FOOOFGroup
from fooof.analysis import get_band_peak_fg

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from info import PATIENTS, N_JOBS, SPEC_PARAM_SETTINGS, ALPHA_RANGE
from stats import gen_random_order, comp_resampling_pval

# FOOOF is causing some warnings about ragged arrays
import warnings
warnings.filterwarnings("ignore")

# analysis/statistical settings
AP_MODE = 'knee' # Specparam aperiodic mode ('fixed' or 'knee')
N_ITER = 100 # number of iterations for permutation test

def main():
    # display progress
    start_time = timer()
    
    # id directories
    dir_input = f'{PROJECT_PATH}/data/ieeg_psd'
    dir_output = f'{PROJECT_PATH}/data/ieeg_stats/permutation_test'
    if not os.path.exists(dir_output): os.makedirs(dir_output)
    
    # initialize 
    dfs = []

    # load channel info
    fname_in = f'{PROJECT_PATH}/data/ieeg_metadata/ieeg_channel_info.pkl'
    chan_info = np.load(fname_in, allow_pickle=True)
    
    # loop through materials
    for material in ['face','word']:
        # loop through memory conditions
        for memory in ['hit','miss']:
            # display progress
            print('---------------------------------------')
            print(f'Condition: {material} - {memory}')
            print('---------------------------------------')
            
            # load spectral parameterization results
            param_pre = FOOOFGroup()
            fname = f"{material}s_{memory}_prestim_params_{AP_MODE}.json"
            param_pre.load(f"{PROJECT_PATH}/data/ieeg_psd_param/{fname}")

            param_post = FOOOFGroup()
            fname = f"{material}s_{memory}_poststim_params_{AP_MODE}.json"
            param_post.load(f"{PROJECT_PATH}/data/ieeg_psd_param/{fname}")
            
            # change NaN to 0 (no detectable alpha peak)
            alpha_pre = get_band_peak_fg(param_pre, ALPHA_RANGE)
            alpha_post = get_band_peak_fg(param_post, ALPHA_RANGE)
            
            # calc change in parameters (exponent and adjusted alpha power)
            exp_diff = param_post.get_params('aperiodic','exponent') - \
                param_pre.get_params('aperiodic','exponent')
            alpha_diff = alpha_post[:,1] - alpha_pre[:,1] 

            # loop through patients
            for patient in PATIENTS:
                # TEMP - skip files that have already been processed
                files_completed = os.listdir(dir_output)
                fname_out = f'stats_{patient}_{material}_{memory}_{AP_MODE}.csv'
                if fname_out in files_completed:
                    print(f'Skipping {fname_out} -  already processed')
                    continue

                # display progress
                start_time_p = timer()
                print(f'\nAnalyzing patient: {patient}' )

                # load subject data (pre/post-stim PSDs)
                fname = f'{patient}_{material}s_{memory}_xxxstim_psd.npz'
                data_pre = np.load(f"{dir_input}/{fname.replace('xxx','pre')}")
                data_post = np.load(f"{dir_input}/{fname.replace('xxx','post')}")
        
                # get parameters for this patient
                exp_diff_pat = exp_diff[chan_info['patient']==patient]
                alpha_diff_pat = alpha_diff[chan_info['patient']==patient]

                # run permutation stats
                df_i = resampling_analysis(data_pre['freq'], data_pre['psd'], 
                                            data_post['psd'], exp_diff_pat,
                                            alpha_diff_pat, AP_MODE, 
                                            ALPHA_RANGE, n_iterations=N_ITER, 
                                            n_jobs=N_JOBS)

                # aggregate
                df_i['patient'] = patient
                df_i['material'] = material
                df_i['memory'] = memory
                dfs.append(df_i)
                
                # save results for patient-material-memory
                fname_out = f'stats_{patient}_{material}_{memory}_{AP_MODE}.csv'
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
        
def resampling_analysis(freq, spectra_pre, spectra_post, exp_diff, alpha_diff, 
                        ap_mode, f_range, n_iterations=1000, n_jobs=-1):
    
    # size up data
    n_trials = spectra_pre.shape[0]
    n_chans = spectra_pre.shape[1]

    # initialize vars
    pval_exp = np.zeros(n_chans)
    sign_exp = np.zeros(n_chans)
    distr_exp = np.zeros([n_chans, n_iterations])
    pval_alpha = np.zeros(n_chans)
    sign_alpha = np.zeros(n_chans)
    distr_alpha = np.zeros([n_chans, n_iterations])
    
    # loop through channels
    for i_chan in range(n_chans):
        # display progress
        start = timer()
        print(f'    Analyzing channel: {i_chan}/{n_chans}')

        
        # shuffle spectra
        order = gen_random_order(n_iterations, int(n_trials*2))
        spectra_0s, spectra_1s = shuffle_spectra(spectra_pre[:, i_chan], 
                                                 spectra_post[:, i_chan], order)

        # parameterize shuffled spectra and compute the difference in exponent
        results = calc_param_change(freq, spectra_0s, spectra_1s, ap_mode, 
                                    f_range, n_jobs=n_jobs)
        distr_exp[i_chan], distr_alpha[i_chan] = results

        # compute p-value
        temp = comp_resampling_pval(distr_exp[i_chan], exp_diff[i_chan])
        pval_exp[i_chan], sign_exp[i_chan] = temp

        temp = comp_resampling_pval(distr_alpha[i_chan], alpha_diff[i_chan])
        pval_alpha[i_chan], sign_alpha[i_chan] = temp
        
        # check if all distr_alpha values are NaN
        if np.isnan(distr_alpha[i_chan]).all():
            pval_alpha[i_chan] = np.nan
            sign_alpha[i_chan] = np.nan
            print('\tall NaN values for alpha power distribution...')

        # time it
        end = timer()
        print(f"\tcomplete in: {end-start:0.0f} seconds")

    # create dataframe of reults
    df = pd.DataFrame({ 'pval_exp' : pval_exp, 
                        'sign_exp' : sign_exp,
                        'pval_alpha' : pval_alpha,
                        'sign_alpha' : sign_alpha})
    df['chan_idx'] = np.arange(n_chans)
        
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

def calc_param_change(freq, spectra_0, spectra_1, ap_mode, f_range, n_jobs=-1):
    # initialize model
    sp_0 = FOOOFGroup(**SPEC_PARAM_SETTINGS, aperiodic_mode=ap_mode, 
                      verbose=False)
    sp_0.set_check_data_mode(False)
    sp_1 = sp_0.copy()

    # fit
    sp_0.fit(freq, spectra_0, n_jobs=n_jobs)
    sp_1.fit(freq, spectra_1, n_jobs=n_jobs)
    
    # calc difference in exponent of shuffled spectra
    exp_diff = sp_1.get_params('aperiodic', 'exponent') - \
               sp_0.get_params('aperiodic', 'exponent')
    
    # calculate change in alpha amplitude
    alpha_0 = get_band_peak_fg(sp_0, f_range)
    alpha_1 = get_band_peak_fg(sp_1, f_range)
    alpha_diff = alpha_1[:,1] - alpha_0[:,1] 
    
    return exp_diff, alpha_diff
        

if __name__ == "__main__":
    main()

