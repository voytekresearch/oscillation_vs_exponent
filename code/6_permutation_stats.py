# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 11:46:53 2021

@author: micha
"""

# Imports
from os.path import join, exists
from os import makedirs
import numpy as np
import pandas as pd
from timeit import default_timer as timer

from fooof import FOOOFGroup
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fg

# Settings
# paths
PROJECT_PATH = 'C:/Users/micha/tilt_vs_fingerprint/'
DIR_STATS = join(PROJECT_PATH, 'data/ieeg_stats/permutation_test')

# If False, analyze channels with reported effect only (Fellner, 2019; n=139)
RUN_ALL_CHANS = False

# signal params
FS = 512 # meg sampling frequency
TMIN = -1.5 # epoch start time

# patient into
PATS = ['pat02','pat04','pat05','pat08','pat10','pat11',
         'pat15','pat16','pat17','pat19','pat20','pat21','pat22'];


# band anlysis
BANDS = Bands({'alpha' : [8, 20]})

# SpecParam settings
PEAK_WIDTH_LIMITS = [2, 20] # default: (0.5, 12.0))
MAX_N_PEAKS = 4 # (default: inf)
MIN_PEAK_HEIGHT = 0 # (default: 0)
PEAK_THRESHOLD =  2 # (default: 2)
AP_MODE = 'knee'

# random permutation settings
N_ITER = 100

# run in parrallel
N_JOBS = 8

def main():
    if RUN_ALL_CHANS:
        run_stats_all_channels()
    else:
        run_stats()
        
    # aggregate statistical results for each condition
    aggregate_stats_by_condition()
    
def run_stats():
    # id directories
    dir_input = join(PROJECT_PATH, 'data/ieeg_psd')
    dir_output = join(PROJECT_PATH, 'data/ieeg_stats/permutation_test')
    if not exists(dir_output):
        makedirs(dir_output)
        
    # load spectral param results
    df = pd.read_pickle(PROJECT_PATH + 'data/results/' + 'df_ols.pkl')

    # loop through materials
    for material in ['word','face']:
        # loop through channels
        current_patient = None
        for ii in range(len(df)):
            # load subject data
            if df.loc[ii,'patient'] == current_patient:
                continue
            else:         
                current_patient = df.loc[ii,'patient']
                fname = '%s_%ss_hit_xxx_psd.npz' %(df.loc[ii,'patient'], material)
                data_pre = np.load(join(dir_input, fname.replace('xxx','prestim')))
                data_post = np.load(join(dir_input, fname.replace('xxx','poststim')))
                
            # get channel data
            spectra_pre = data_pre['psd'][:, df.loc[ii,'chan_idx']]
            spectra_post = data_post['psd'][:, df.loc[ii,'chan_idx']]
            freq = data_pre['freq']
            diff_alpha = df.loc[ii, f'diff_alpha_{material[0]}']
            diff_exp = df.loc[ii, f'diff_exp_{material[0]}']
            
            # run stats (display progress)
            print('running channel %d / %d: ...' %(ii, len(df)))
            start = timer()
            results = rand_perm_stats(freq, spectra_pre, spectra_post, 
                                      diff_exp, diff_alpha, AP_MODE, BANDS)
            end = timer()
            print('\t%0.1f seconds' %(end-start))

            # save results         
            fname_out = '%s_chan%s_permutest' %(df.loc[ii,'patient'], 
                                                df.loc[ii,'chan_idx'])
            pval_exp, sign_exp, distr_exp, \
            pval_alpha, sign_alpha, distr_alpha = results
            np.savez(join(dir_output, fname_out), pval_exp=pval_exp, 
                     sign_exp=sign_exp, distr_exp=distr_exp, 
                     diff_exp=diff_exp, pval_alpha=pval_alpha, 
                     sign_alpha=sign_alpha, diff_alpha=diff_alpha)


def rand_perm_stats(freq, spectra_pre, spectra_post, exp_diff, alpha_diff, 
                    ap_mode, bands, n_iterations=1000, n_jobs=1):            
    # shuffle spectra
    order = gen_random_order(n_iterations, int(spectra_pre.shape[0]*2))
    spectra_0s, spectra_1s = shuffle_spectra(spectra_pre, spectra_post, order)
    
    # parameterize shuffled spectra and compute the difference in params
    results = calc_param_change(freq, spectra_0s, spectra_1s, ap_mode, bands, n_jobs=n_jobs)
    distr_exp, distr_alpha = results
    
    # comp p-value
    pval_exp, sign_exp = comp_rand_perm_p_val(distr_exp, exp_diff)
    pval_alpha, sign_alpha = comp_rand_perm_p_val(distr_alpha, alpha_diff)        
    
    return pval_exp, sign_exp, distr_exp, pval_alpha, sign_alpha, distr_alpha
  
    
def run_stats_all_channels():
    # id directories
    dir_input = join(PROJECT_PATH, 'data/ieeg_psd')
    dir_output = join(PROJECT_PATH, 'data/ieeg_stats/permutation_test_allchans')
    if not exists(dir_output):
        makedirs(dir_output)
    
    # load channel info
    chan_info = np.load(join(PROJECT_PATH, 'data/ieeg_metadata', 
                             'ieeg_channel_info.pkl'), allow_pickle=True)
    
    # loop through materials
    for material in ['word','face']:
        # display progress
        print('---------------------------------------')
        print('Condition: %s' %material)
        print('---------------------------------------')
        
        # load spectral results
        param_pre = FOOOFGroup()
        param_pre.load(join(PROJECT_PATH, 'data/ieeg_psd_param', 
                            'psd_pre_%s_params_%s.json' %(material, AP_MODE)))
        param_post = FOOOFGroup()
        param_post.load(join(PROJECT_PATH, 'data/ieeg_psd_param', 
                             'psd_post_%s_params_%s.json' %(material, AP_MODE)))
        
        # change NaN to 0
        alpha_pre = get_band_peak_fg(param_pre, BANDS.alpha)
        alpha_post = get_band_peak_fg(param_post, BANDS.alpha)
        alpha_pre[np.isnan(alpha_pre)] == 0
        alpha_post[np.isnan(alpha_post)] == 0
        
        # calc change in parameters
        exp_diff = param_post.get_params('aperiodic','exponent') - param_pre.get_params('aperiodic','exponent')
        alpha_diff = alpha_post[:,1] - alpha_pre[:,1] 

        # loop through patients
        for i_pat, patient in enumerate(PATS):
            # display progress
            print('Analyzing: %s - %s' %(patient, material))
            
            # load subject data
            fname = '%s_%ss_hit_xxx_psd.npz' %(patient, material)
            data_pre = np.load(join(dir_input, fname.replace('xxx','prestim')))
            data_post = np.load(join(dir_input, fname.replace('xxx','poststim')))
    
            # run stats
            exp_diff_pat = exp_diff[chan_info['patient']==patient]
            alpha_diff_pat = alpha_diff[chan_info['patient']==patient]
            results = rand_perm_stats_sub(data_pre['freq'], data_pre['psd'], 
                                          data_post['psd'], exp_diff_pat,
                                          alpha_diff_pat, AP_MODE, BANDS,
                                          n_iterations=N_ITER, n_jobs=N_JOBS)

            pval_exp, sign_exp, distr_exp, \
            pval_alpha, sign_alpha, distr_alpha = results
            
            # save results
            # fname_out = '%s_%s_%s_diff_exp_randperm' %(patient, material, ap_mode)
            fname_out = '%s_%s_%s_permutest' %(patient, material, AP_MODE)
            np.savez(join(dir_output, fname_out), pval_exp=pval_exp, 
                     sign_exp=sign_exp, distr_exp=distr_exp, 
                     diff_exp=exp_diff_pat, pval_alpha=pval_alpha, 
                     sign_alpha=sign_alpha, diff_alpha=alpha_diff_pat)
        
def rand_perm_stats_sub(freq, spectra_pre, spectra_post, exp_diff, alpha_diff, 
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

        #  parameterize shuffled spectra and 
        # compute the difference in exponent
        results = calc_param_change(freq, spectra_0s, spectra_1s, ap_mode, bands, n_jobs=n_jobs)
        distr_exp[i_chan], distr_alpha[i_chan] = results

        # comp p-value
        pval_exp[i_chan], sign_exp[i_chan] = comp_rand_perm_p_val(distr_exp[i_chan], 
                                                           exp_diff[i_chan])
        pval_alpha[i_chan], sign_alpha[i_chan] = comp_rand_perm_p_val(distr_alpha[i_chan], 
                                                           alpha_diff[i_chan])        
        # time it
        end = timer()
        print('channel %d / %d: %0.1f seconds' %(i_chan+1, n_chans, end-start))
        
    return pval_exp, sign_exp, distr_exp, pval_alpha, sign_alpha, distr_alpha

def gen_random_order(n_iterations, length):
    # generate pseudo-random order for trials
    order = np.zeros([n_iterations, length])
    for i_iter in range(n_iterations):
        order[i_iter] = np.random.permutation(np.linspace(0, length-1, length))
    order = order.astype(int)
    
    return order
    
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
    
    return spectra_0s, spectra_1s

def calc_param_change(freq, spectra_0, spectra_1, ap_mode, bands, n_jobs=1):
    # initialize model
    sp_0 = FOOOFGroup(peak_width_limits = PEAK_WIDTH_LIMITS,
                    max_n_peaks = MAX_N_PEAKS,
                    min_peak_height = MIN_PEAK_HEIGHT,
                    peak_threshold=PEAK_THRESHOLD,
                    aperiodic_mode=ap_mode, verbose=False)
    sp_1 = sp_0.copy()

    # ignore NaN
    # sp_0.set_check_data_mode(False)
    # sp_1.set_check_data_mode(False)

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

def comp_rand_perm_p_val(distribution, value):
    n_iterations = np.size(distribution)
    n_less = np.sum(distribution < value)
    n_more = np.sum(distribution > value)
    
    # calc 2-sided p value
    p_value = np.min([n_less, n_more]) / n_iterations * 2
    
    # determine direction of effect
    if n_less < n_more: 
        sign = -1
    elif n_less > n_more: 
        sign = 1
    elif n_less == n_more: 
        sign = 0
        
    return p_value, sign

def aggregate_stats_by_condition():
# load stats and aggregate group results

    # loop through both materials
    for material in ['word', 'face']:
        # initialize
        pval_exp = []
        pval_alpha = []
        sign_exp = []
        sign_alpha = []
        
        for i_pat, patient in enumerate(PATS):
            fname_in = join(DIR_STATS, '%s_%s_%s_permutest.npz' %(patient, material, AP_MODE))
            data_in = np.load(fname_in)
            pval_alpha = np.append(pval_alpha, data_in['pval_alpha'])
            sign_alpha = np.append(sign_alpha, data_in['sign_alpha'])
            pval_exp = np.append(pval_exp, data_in['pval_exp'])
            sign_exp = np.append(sign_exp, data_in['sign_exp'])
        
        np.savez(join(DIR_STATS, 'group_stats_%s_%s' %(material, AP_MODE)),
                pval_alpha=pval_alpha, pval_exp=pval_exp, sign_exp=sign_exp,
                sign_alpha=sign_alpha)
        

if __name__ == "__main__":
    main()

