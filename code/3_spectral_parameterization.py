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
from fooof import FOOOFGroup
from fooof.utils.data import interpolate_spectrum

# Settings
LINE_NOISE_RANGE = [45,55] # freq range to interpolate
RUN_TFR = False # run TFR parameterization

# SpecParam hyperparameters
N_JOBS = -1 # number of jobs for parallel processing
SPEC_PARAM_SETTINGS = {
    'peak_width_limits' :   [4, 20], # default: (0.5, 12.0)) - recommends at least frequency resolution * 2
    'min_peak_height'   :   0.1, 
    'max_n_peaks'       :   4, # (default: inf)
    'peak_threshold'    :   2.0} # (default: 2.0)

def main():
    
    # parameterize PSDs
    # parameterize_psd()
    param_group_psd_results()

    # parameterize TFRs
    if RUN_TFR:
        # parameterize_tfr()
        param_group_tfr_results()
    
def param_group_psd_results():
    # identify / create directories
    dir_input = f"{PROJECT_PATH}/data/ieeg_spectral_results"
    dir_output = f"{PROJECT_PATH}/data/ieeg_psd_param"
    if not os.path.exists(dir_output): 
        os.makedirs(f"{dir_output}/fooof_reports")
    
    # loop through conditions
    files = ['psd_words_hit_prestim.npz', 'psd_words_hit_poststim.npz',
             'psd_faces_hit_prestim.npz', 'psd_faces_hit_poststim.npz',
             'psd_words_miss_prestim.npz', 'psd_words_miss_poststim.npz',
             'psd_faces_miss_prestim.npz', 'psd_faces_miss_poststim.npz']
    conditions = ['words_hit_prestim', 'words_hit_poststim', 
                  'faces_hit_prestim', 'faces_hit_poststim',
                  'words_miss_prestim', 'words_miss_poststim', 
                  'faces_miss_prestim', 'faces_miss_poststim']
    
    for file, cond in zip(files, conditions):
        # load results for condition
        data_in =  np.load(f"{dir_input}/{file}")
        spectra_raw = data_in['spectra']
        freq = data_in['freq']
        
        # interpolate psd for frequency range that includes line noise
        spectra = np.zeros_like(spectra_raw)
        for chan in range(len(spectra)):
            _, spectra[chan] = interpolate_spectrum(freq, spectra_raw[chan], 
                                                    LINE_NOISE_RANGE)
        
        # parameterize (fit both with and without knee parametere)
        for ap_mode in ['fixed', 'knee']:
            fg = FOOOFGroup(**SPEC_PARAM_SETTINGS, aperiodic_mode=ap_mode, verbose=False)
            fg.set_check_data_mode(False)
            fg.fit(freq, spectra, n_jobs=N_JOBS)
            
            # save results 
            fname_out = '%s_params_%s' %(cond, ap_mode)
            fg.save(f"{dir_output}/{fname_out}", save_results=True, 
                    save_settings=True)
            fg.save_report(f"{dir_output}/fooof_reports/{fname_out}")


def param_group_tfr_results():

    # identify / create directories
    dir_input = f"{PROJECT_PATH}/data/ieeg_spectral_results"
    dir_output = f"{PROJECT_PATH}/data/ieeg_tfr_param"
    if not os.path.exists(dir_output): 
        os.makedirs(f"{dir_output}/fooof_reports")
        
    # loop through conditions
    conditions = ['words_hit', 'faces_hit','words_miss', 'faces_miss']
    for cond in conditions:
        # load group TFR results
        data_in = np.load(f"{dir_input}/'tfr_{cond}_hit_multitaper.npz")
        freq = data_in['freq']
        tfr_in = data_in['tfr']
        
        # downsample tfr
        tfr_ds = downsample_tfr(tfr_in)
        
        # reshape TFR for parameterization
        tfr = tfr_ds.reshape([tfr_ds.shape[0]*tfr_ds.shape[1],tfr_ds.shape[2]])
        
        # parameterize (fit both with and without knee parametere)
        for ap_mode in ['fixed', 'knee']:
            fg = FOOOFGroup(**SPEC_PARAM_SETTINGS, aperiodic_mode=ap_mode, verbose=False)
            fg.set_check_data_mode(False)
            fg.fit(freq, tfr, n_jobs=N_JOBS)
            
            # save results and report
            fname_out = 'tfr_%s_params_%s' %(cond, ap_mode)
            fg.save(f"{dir_output}/{fname_out}", save_results=True, 
                    save_settings=True)
            fg.save_report(f"{dir_output}/fooof_reports/{fname_out}")
        

# def parameterize_psd():
#     # identify / create directories
#     dir_input = f"{PROJECT_PATH}/data/ieeg_psd"
#     dir_output = f"{PROJECT_PATH}/data/ieeg_psd_param"
#     if not os.path.exists(dir_output): 
#         os.makedirs(f"{dir_output}/fooof_reports")
    
#     # load each file
#     for fname in os.path.listdir(dir_input):
        
#         # display progress
#         print('\n__________Analyzing: %s ____________________\n' %fname)
        
#         # load psd
#         data_in = np.load(join(dir_input, fname))
#         psd_in = data_in['psd']
#         freq = data_in['freq']
        
#         # average over trials
#         psd_mean = np.nanmean(psd_in, axis=0)
            
#         # interpolate psd for frequency range that includes line noise
#         spectra = np.zeros_like(psd_mean)
#         for chan in range(len(spectra)):
#             _, spectra[chan] = interpolate_spectrum(freq, psd_mean[chan], 
#                                                     LINE_NOISE_RANGE)

#         # parameterize (fit both with and without knee parametere)
#         for ap_mode in ['fixed', 'knee']:
#             fg = FOOOFGroup(**SPEC_PARAM_SETTINGS, aperiodic_mode=ap_mode, verbose=False)
#             fg.set_check_data_mode(False)
#             fg.fit(freq, spectra, n_jobs=N_JOBS)
            
#             # save results 
#             fname_out = fname.replace('.npz','_param_%s' %ap_mode)
#             fg.save(f"{dir_output}/{fname_out}", save_results=True, 
#                     save_settings=True)
#             fg.save_report(f"{dir_output}/fooof_reports/{fname_out}")
     
                
# def parameterize_tfr():

#     # identify / create directories
#     dir_input = f"{PROJECT_PATH}/data/ieeg_tfr"
#     dir_output = f"{PROJECT_PATH}/data/ieeg_tfr_param"
#     if not os.path.exists(dir_output): 
#         os.makedirs(f"{dir_output}/fooof_reports")
    
#     # load each file
#     for fname in os.path.listdir(dir_input):
#         # use multitaper decomposition 
#         fparts = fname.split('_')
#         if not fparts[-1] == 'multitaper.npz': continue
        
#         # display progress
#         print('Analyzing: %s' %fname)
        
#         # load tfr
#         data_in = np.load(f"{dir_input}/{fname}")
#         tfr_in = data_in['tfr']
#         freq = data_in['freq']
        
#         # average over trials
#         tfr_mean = np.squeeze(np.mean(tfr_in, axis=0))
        
#         # downsample tfr
#         tfr = downsample_tfr(tfr_mean)
        
#         # parameterize
#         for ap_mode in ['fixed', 'knee']:
#             fg = FOOOFGroup(**SPEC_PARAM_SETTINGS, aperiodic_mode=ap_mode, verbose=False)
#             fg.set_check_data_mode(False)
#             fg.fit(freq, tfr, n_jobs=N_JOBS)
            
#             # save results and report
#             fname_out = fname.replace('.npz','_param_%s' %ap_mode)
#             fg.save(f"{dir_output}/{fname_out}", save_results=True, 
#                     save_settings=True)
#             fg.save_report(f"{dir_output}/'fooof_reports/{fname_out}")
        
        
def downsample_tfr(tfr, n=2**7):
    # downsample tfr in time dimension
    # n: number of samples after downsampling
    
    if np.ndim(tfr) == 2:
        step = int(np.floor(tfr.shape[1] / n))
        tfr_ds = tfr[:, np.arange(1, tfr.shape[1], step)]
    elif np.ndim(tfr) == 3:
        step = int(np.floor(tfr.shape[1] / n))
        tfr_ds = tfr[:, np.arange(1, tfr.shape[1], step), :]
    
    return tfr_ds

     
        
if __name__ == "__main__":
    main()