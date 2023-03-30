# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 17:35:00 2021

@author: micha

Data Repo: https://osf.io/3csku/
Associated Paper: https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000403

 
This script executes the primary time-frequnecy analyses. 
The power spectral density (PSD) is computed for each epoch, the pre-stimulus 
time window, and the post-stimulus time window. 
The time-frequnecy representation of power (TFR) is computed for the epoch 
using either Morlet wavelet or multitapers. 

"""
# Imports

from os.path import join, exists
from os import mkdir, listdir
import numpy as np
from mne import read_epochs, create_info, EpochsArray
from mne.time_frequency import psd_multitaper, tfr_multitaper, tfr_morlet

# Parameters
PROJECT_PATH = 'C:/Users/micha/projects/oscillation_vs_exponent/'
PATIENTS = ['pat02','pat04','pat05','pat08','pat10','pat11','pat15','pat16',
            'pat17','pat19','pat20','pat21','pat22']

# parameters for psd analysis
N_JOBS = -1 # number of jobs to run in parallel (-1 = use all available cores)
BANDWIDTH = 2 # multitaper bandwidth - frequencies at ± half-bandwidth are smoothed together
TIME_RANGE = np.array([[-1.0, 1.0],    # epoch
                       [-1.0, 0.0],    # pre-stim
                       [0.0, 1.0]])    # post-stim
TIME_RANGE_LABELS = np.array(['epoch',
                              'prestim',
                              'poststim'])

# parameters for tfr analysis
RUN_TFR = True # set to False to skip tfr analysis
TFR_METHOD = ['multitaper']

def main():
    # identify / create directories
    dir_input = join(PROJECT_PATH, 'data/ieeg_epochs')
    dir_psd = join(PROJECT_PATH, 'data/ieeg_psd')
    dir_tfr = join(PROJECT_PATH, 'data/ieeg_tfr')
    dir_results = join(PROJECT_PATH, 'data/ieeg_spectral_results')
    if not exists(dir_psd): mkdir(dir_psd)
    if not exists(dir_tfr): mkdir(dir_tfr)
    if not exists(dir_results): mkdir(dir_results)
    
    # for each fif file
    for fname in listdir(dir_input):
        # display progress
        print(f"\nAnalyzing: {fname}")
        
        # load eeg data
        epochs = read_epochs(join(dir_input, fname), verbose=False)
        
        # compute power spectral density
        comp_psd(epochs, fname, dir_psd)
    
        # compute time-frequency representation of power,
        # for each trial/channel
        if RUN_TFR:
            compute_channel_tfr(epochs, fname, dir_tfr)
    
    # aggregate psd results. average over trials
    aggregate_spectra(dir_psd, dir_results)
    
    # aggregate tfr results. average over trials
    if RUN_TFR:
        aggregate_tfr(dir_tfr, dir_results)
    
def comp_psd(epochs, fname, dir_output):
    '''
    This function takes an MNE epochsArray and computes the power spectral 
    density (PSD). Spectra are calculated for several specified time windows.
    '''
    
    # for the pre-stimulus and post-stimulus condition
    for label, time_range in zip(TIME_RANGE_LABELS, TIME_RANGE):
        
        # calculate PSD
        psd, freq = psd_multitaper(epochs, tmin=time_range[0], tmax=time_range[1],
                                    bandwidth=BANDWIDTH, n_jobs=N_JOBS, 
                                    verbose=False)
        
        # save power results
        fname_out = str.replace(fname, '_epo.fif', '_%s_psd' %(label))
        np.savez(join(dir_output, fname_out), 
                 psd = psd, 
                 freq = freq)


def compute_channel_tfr(epochs, fname, dir_output):
    '''
    This function takes an MNE epochsArray and computes the time-frequency
    representatoin of power for each channel sequentially, saving the results
    for each channel seperately. 
    '''
    
    # get single channel epochs, and compute TFR for each channel
    for channel in range(len(epochs.info['ch_names'])):
        # get single channle epochs
        epochs_chan = get_single_channel_epochs(epochs, channel)
        
        # skip channels with no data or less than 2 trials
        # note: some channels in dataset contain all NaN; only 1 trial causes error
        if epochs_chan == 'no data': continue
        
        # run time-frequency analysis
        for method in TFR_METHOD:
            tfr, freq = compute_tfr(epochs_chan, method=method)
            
            # save time-frequency results
            fname_out = str.replace(fname, '_epo.fif', \
                                    '_chan%s_tfr_%s' %(channel, method))
            np.savez(join(dir_output, fname_out), 
                     tfr=tfr, freq=freq, time=epochs_chan.times)


def get_single_channel_epochs(epochs, channel):
    '''
    This function takes an MNE epochsArray and creates an new epochsArray 
    containing the data for single specified channel. 
    Rejected trials are removed from epochsArray.
    '''
    
    # get data for channel
    lfp = epochs.get_data(picks=channel)

    # check data contains at least 2 trials
    # note: some channels in dataset contain all NaN; only 1 trial causes error
    if np.isnan(lfp).all() or len(lfp) < 2:
        epochs_chan = 'no data'
        
    else:
        # remove missing/rejected data trials
        lfp = lfp[~np.isnan(lfp[:, 0, 0])]
    
        #create MNE epochs array
        info = create_info(np.shape(lfp)[1], epochs.info['sfreq'], \
                               ch_types='eeg')
        epochs_chan = EpochsArray(lfp, info, tmin=epochs.tmin, verbose=False)
    
    return epochs_chan


def compute_tfr(epochs, method='multitaper', average_trials=False):
    '''
    This function takes an MNE epochsArray and computes the time-frequency
    representatoin of power using either multitapers or wavelets. 
    Time-frequnecy parameters are set to replicate Fellner et al.
    Due to memory demands, this function should be run on single-channel data, 
    or results can be averaged across trials.
    '''
    
    if method=='multitaper':
        # set paramters for TF decomposition
        freq = np.logspace(*np.log10([2,100]),128)
        n_cycles = freq / (1/0.3) # 300 ms - as published
        time_bandwidth = 10 * 0.3 # 10 Hz (when T=0.3 sec) - as published
        
        # TF decomposition using multitapers
        tfr = tfr_multitaper(epochs, freqs=freq, n_cycles=n_cycles, 
                             time_bandwidth=time_bandwidth,
                             use_fft=True, return_itc=False, 
                             average=average_trials, verbose=False,
                             n_jobs=N_JOBS)
    
    elif (method=='morlet'):
        # set paramters for TF decomposition
        freq = np.logspace(*np.log10([2,100]),128)
        n_cycles = 5 # 5 cyces - as published
        
        # TF decomposition using morlet wavelets
        tfr = tfr_morlet(epochs, freqs=freq, n_cycles=n_cycles, 
                             use_fft=True, return_itc=False, 
                             average=average_trials, verbose=False,
                             n_jobs=N_JOBS)

    elif method=='morlet_adpt':
        # set paramters for TF decomposition
        freq = np.logspace(*np.log10([2,100]),128)
        n_cycles = freq / (1/0.3) # 300 ms
        
        # TF decomposition using morlet wavelets
        tfr = tfr_morlet(epochs, freqs=freq, n_cycles=n_cycles, 
                             use_fft=True, return_itc=False, 
                             average=average_trials, verbose=False,
                             n_jobs=N_JOBS)
    
    # define variables to return
    tfr = tfr.data

    return tfr, freq

def aggregate_spectra(dir_input, dir_output):
    '''
    This function aggregates the PSD results across files, for each condition. 
    Trial results are averaged (median) for each channel.
    '''
    
    # load frequency vector
    files = listdir(dir_input)
    data_in = np.load(join(dir_input, files[1]))
    freq = data_in['freq']
    
    # aggregate psd data for each condition
    conditions = ['words_hit_prestim', 'words_hit_poststim', 'faces_hit_prestim', 'faces_hit_poststim',
                  'words_miss_prestim', 'words_miss_poststim', 'faces_miss_prestim', 'faces_miss_poststim']
    for cond in conditions:    
        # create placeholder for output data
        spectra = np.zeros(len(freq))
        for patient in PATIENTS:    
            # load psd data               
            data_in = np.load(join(dir_input, '%s_%s_psd.npz' %(patient, cond)))
            spectra = np.vstack([spectra, np.nanmedian(data_in['psd'], axis=0)])

        # remove place-holder and save results
        spectra = spectra[1:]
        fname_out = 'psd_%s' %cond
        np.savez(join(dir_output, fname_out), freq=freq, spectra=spectra)
        
def aggregate_tfr(dir_input, dir_output):
    '''
    This function aggregates the tfr results across files, for each condition. 
    Trial results are averaged (median) for each channel.
    '''
    
    # load frequency vector
    files = listdir(dir_input)
    data_in = np.load(join(dir_input, files[0]))
    freq = data_in['freq']
    time = data_in['time']
    
    # load channel meta data
    meta = np.load(join(PROJECT_PATH, 'data/ieeg_metadata', 'ieeg_channel_info.pkl'),
                   allow_pickle=True)
    
    # aggregate psd data for each condition
    for condition in ['words_hit', 'faces_hit', 'words_miss', 'faces_miss']:
        for method in TFR_METHOD:
            # create array for output data
            tfr = np.zeros([len(meta), len(time), len(freq)])
            
            # loop through rows of metadata
            for ii in range(len(meta)):
                fname_in = '%s_%s_chan%d_tfr_%s.npz' %(meta['patient'][ii], condition, meta['chan_idx'][ii], method)
                # skip missing channels
                if not fname_in in files: continue
            
                # load tfr data               
                data_in = np.load(join(dir_input, fname_in))
                tfr[ii] = np.squeeze(np.nanmedian(data_in['tfr'], axis=0).T)

        #  save results
        fname_out = 'tfr_%s_%s' %(condition, method)
        np.savez(join(dir_output, fname_out), freq=freq, time=time, tfr=tfr)


if __name__ == "__main__":
    main()
    
    