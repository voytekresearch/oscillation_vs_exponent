# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 11:46:53 2021

@author: micha
"""

# Set paths
PROJECT_PATH = 'C:/Users/micha/projects/oscillation_vs_exponent/'

# Imports - general
import os
import numpy as np
import pandas as pd
from timeit import default_timer as timer

from fooof import FOOOFGroup
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fg

# Imports - custom
from stats import gen_random_order, comp_resampling_pval

# analysis/statistical settings
AP_MODE = 'knee' # Specparam setting, 'fixed' or 'knee' - knee is recommended for this dataset
N_ITER = 100 # number of iterations for permutation test
N_JOBS = -1 # run in parrallel

# dataset details
FS = 512 # meg sampling frequency
TMIN = -1.5 # epoch start time
PATS = ['pat02','pat04','pat05','pat08','pat10','pat11',
         'pat15','pat16','pat17','pat19','pat20','pat21','pat22'];

# analysis parameters used to generate results
BANDS = Bands({'alpha' : [8, 20]})
SPEC_PARAM_SETTINGS = {
    'peak_width_limits' :   [4, 20], # default: (0.5, 12.0)) - recommends at least frequency resolution * 2
    'min_peak_height'   :   0.1, 
    'max_n_peaks'       :   4, # (default: inf)
    'peak_threshold'    :   2.0} # (default: 2.0)


def main():
    # id directories
    dir_input = f'{PROJECT_PATH}/data/ieeg_psd'
    dir_output = f'{PROJECT_PATH}/data/ieeg_stats/permutation_test'
    if not os.path.exists(dir_output): os.makedirs(dir_output)
    
    # load channel info
    chan_info = np.load(f'{PROJECT_PATH}/data/ieeg_metadata/ieeg_channel_info.pkl',
        allow_pickle=True)
    
    # loop through materials
    for material in ['word','face']:
        # display progress
        print('---------------------------------------')
        print('Condition: %s' %material)
        print('---------------------------------------')
        
        # load spectral parameterization results
        param_pre = FOOOFGroup()
        param_pre.load(f"{PROJECT_PATH}/data/ieeg_psd_param/psd_pre_{material}_params_{AP_MODE}.json")
        param_post = FOOOFGroup()
        param_post.load(f"{PROJECT_PATH}/'data/ieeg_psd_param/psd_post_{material}_params_{AP_MODE}.json")
        
        # change NaN to 0 (no detectable alpha peak)
        alpha_pre = get_band_peak_fg(param_pre, BANDS.alpha)
        alpha_post = get_band_peak_fg(param_post, BANDS.alpha)
        alpha_pre[np.isnan(alpha_pre)] == 0
        alpha_post[np.isnan(alpha_post)] == 0
        
        # calc change in parameters (exponent and adjusted alpha power)
        exp_diff = param_post.get_params('aperiodic','exponent') - \
            param_pre.get_params('aperiodic','exponent')
        alpha_diff = alpha_post[:,1] - alpha_pre[:,1] 

        # loop through patients
        for i_pat, patient in enumerate(PATS):
            # display progress
            print('Analyzing: %s - %s' %(patient, material))
            
            # load subject data (pre/post-stim PSDs)
            fname = '%s_%ss_hit_xxx_psd.npz' %(patient, material)
            data_pre = np.load(f"{dir_input}/{fname.replace('xxx','prestim')}")
            data_post = np.load(f"{dir_input}/{fname.replace('xxx','poststim')}")
    
            # run permutation stats
            exp_diff_pat = exp_diff[chan_info['patient']==patient]
            alpha_diff_pat = alpha_diff[chan_info['patient']==patient]
            results = resampling_analysis(data_pre['freq'], data_pre['psd'], 
                                          data_post['psd'], exp_diff_pat,
                                          alpha_diff_pat, AP_MODE, BANDS,
                                          n_iterations=N_ITER, n_jobs=N_JOBS)

            pval_exp, sign_exp, distr_exp, \
            pval_alpha, sign_alpha, distr_alpha = results
            
            # save results
            # fname_out = '%s_%s_%s_diff_exp_randperm' %(patient, material, ap_mode)
            fname_out = f'{patient}_{material}_{AP_MODE}_permutest'
            np.savez(f"{dir_output}/{fname_out}", pval_exp=pval_exp, 
                     sign_exp=sign_exp, distr_exp=distr_exp, 
                     diff_exp=exp_diff_pat, pval_alpha=pval_alpha, 
                     sign_alpha=sign_alpha, diff_alpha=alpha_diff_pat)
            
    # aggregate statistical results for each condition
    aggregate_stats_by_condition(dir_output)    

        
def resampling_analysis(freq, spectra_pre, spectra_post, exp_diff, alpha_diff, 
                        ap_mode, bands, n_iterations=1000, n_jobs=1):
    
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
        
        # shuffle spectra
        order = gen_random_order(n_iterations, int(n_trials*2))
        spectra_0s, spectra_1s = shuffle_spectra(spectra_pre[:, i_chan], 
                                                 spectra_post[:, i_chan], order)

        # parameterize shuffled spectra and compute the difference in exponent
        results = calc_param_change(freq, spectra_0s, spectra_1s, ap_mode, bands, n_jobs=n_jobs)
        distr_exp[i_chan], distr_alpha[i_chan] = results

        # comp p-value
        pval_exp[i_chan], sign_exp[i_chan] = comp_resampling_pval(distr_exp[i_chan], 
                                                           exp_diff[i_chan])
        pval_alpha[i_chan], sign_alpha[i_chan] = comp_resampling_pval(distr_alpha[i_chan], 
                                                           alpha_diff[i_chan])        
        # time it
        end = timer()
        print('channel %d / %d: %0.1f seconds' %(i_chan+1, n_chans, end-start))
        
    return pval_exp, sign_exp, distr_exp, pval_alpha, sign_alpha, distr_alpha
    
def shuffle_spectra(spectra_0, spectra_1, order):
    # concatenate 2 groups of spectra
    n_spectra = spectra_0.shape[0]
    spectra = np.concatenate([spectra_0, spectra_1])

    # shuffle spectra and average
    spectra_0s = np.zeros([order.shape[0], spectra.shape[1]])
    spectra_1s = spectra_0s.copy()
    for i_iter in range(order.shape[0]):
        spectra_0s[i_iter] = np.nanmean(spectra[order[i_iter, :n_spectra]], 0)
        spectra_1s[i_iter] = np.nanmean(spectra[order[i_iter, n_spectra:]], 0)
        # spectra_0s[i_iter] = np.nanmedian(spectra[order[i_iter, :n_spectra]], 0)
        # spectra_1s[i_iter] = np.nanmedian(spectra[order[i_iter, n_spectra:]], 0)

    return spectra_0s, spectra_1s

def calc_param_change(freq, spectra_0, spectra_1, ap_mode, bands, n_jobs=1):
    # initialize model
    sp_0 = FOOOFGroup(*SPEC_PARAM_SETTINGS, aperiodic_mode=AP_MODE, verbose=False)
    sp_1 = sp_0.copy()

    # fit
    sp_0.fit(freq, spectra_0, n_jobs=n_jobs)
    sp_1.fit(freq, spectra_1, n_jobs=n_jobs)
    
    # calc difference in exponent of shuffled spectra
    exp_diff = sp_1.get_params('aperiodic', 'exponent') - \
               sp_0.get_params('aperiodic', 'exponent')
    
    # calculate change in alpha amplitude
    alpha_0 = get_band_peak_fg(sp_0, bands.alpha)
    alpha_1 = get_band_peak_fg(sp_1, bands.alpha)
    alpha_0[np.isnan(alpha_0)] == 0
    alpha_1[np.isnan(alpha_1)] == 0
    alpha_diff = alpha_1[:,1] - alpha_0[:,1] 
    
    return exp_diff, alpha_diff

def aggregate_stats_by_condition(path_out):
# load stats and aggregate group results

    # loop through both materials
    for material in ['word', 'face']:
        # initialize
        pval_exp = []
        pval_alpha = []
        sign_exp = []
        sign_alpha = []
        
        for patient in PATS:
            fname_in = f"{path_out}/{patient}_{material}_{AP_MODE}_permutest.npz"
            data_in = np.load(fname_in)
            pval_alpha = np.append(pval_alpha, data_in['pval_alpha'])
            sign_alpha = np.append(sign_alpha, data_in['sign_alpha'])
            pval_exp = np.append(pval_exp, data_in['pval_exp'])
            sign_exp = np.append(sign_exp, data_in['sign_exp'])
        
        np.savez(f"{path_out}/group_stats_{material}_{AP_MODE}",
                pval_alpha=pval_alpha, pval_exp=pval_exp, sign_exp=sign_exp,
                sign_alpha=sign_alpha)
        

if __name__ == "__main__":
    main()

