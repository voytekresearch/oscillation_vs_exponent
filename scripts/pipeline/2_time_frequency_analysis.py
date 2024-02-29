"""
This script executes the primary time-frequnecy analyses. The power spectral 
density (PSD) is computed for each epoch, the pre-stimulus time window, and the 
post-stimulus time window. The time-frequnecy representation of power (TFR) is 
also computed for each epoch using the multitaper method. 

"""

# Imports - standard
from os.path import join, exists
from os import mkdir, listdir
import numpy as np
from mne import read_epochs
from mne.time_frequency import tfr_multitaper
from time import time as timer

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from info import PATIENTS
from settings import N_JOBS, N_TFR_SAMPLES, EPOCH_TIMES, EPOCH_LABELS
from utils import hour_min_sec
from tfr_utils import crop_tfr

# Settings
RUN_TFR = True # set to False to skip tfr analysis (long run time)
PSD_BANDWIDTH = 2 # frequencies at ± bandwidth are smoothed 


def main():
    # identify / create directories
    dir_input = join(PROJECT_PATH, 'data/ieeg_epochs')
    dir_psd = join(PROJECT_PATH, 'data/ieeg_psd')
    dir_tfr = join(PROJECT_PATH, 'data/ieeg_tfr')
    dir_results = join(PROJECT_PATH, 'data/ieeg_spectral_results')
    if not exists(dir_psd): mkdir(dir_psd)
    if not exists(dir_tfr): mkdir(dir_tfr)
    if not exists(dir_results): mkdir(dir_results)
    
    # display progress
    t_start = timer()

    # for each fif file
    files = listdir(dir_input)
    for ii, fname in enumerate(files):

        # display progress
        t_start_f = timer()
        print(f"\nAnalyzing file {ii}/{len(files)}")
        print(f"\tfilename: \t{fname}")
        
        # load eeg data
        epochs = read_epochs(join(dir_input, fname), verbose=False)
        print(f"\tchannels: \t{len(epochs.info['ch_names'])}")
        
        # compute power spectral density
        comp_psd(epochs, fname, dir_psd)
    
        # compute time-frequency representation of power,
        # for each trial/channel
        if RUN_TFR:
            compute_channel_tfr(epochs, fname, dir_tfr)
        
        # display progress
        hour, min, sec = hour_min_sec(timer() - t_start_f)
        print(f"\tanalysis time: \t{hour} hour, {min} min, and {sec :0.1f} s")

    # aggregate psd results. average over trials
    aggregate_spectra(dir_psd, dir_results)
    
    # aggregate tfr results. average over trials
    if RUN_TFR:
        aggregate_tfr(dir_tfr, dir_results)

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\n\nTotal analysis time: {hour} hour, {min} min, and {sec :0.1f} s")
    
def comp_psd(epochs, fname, dir_output):
    '''
    This function takes an MNE epochsArray and computes the power spectral 
    density (PSD). Spectra are calculated for several specified time windows.
    '''
    
    # for the pre-stimulus and post-stimulus condition
    for label, time_range in zip(EPOCH_LABELS, EPOCH_TIMES):
        
        # calculate PSD
        results = epochs.compute_psd(tmin=time_range[0], tmax=time_range[1], 
                                       bandwidth=PSD_BANDWIDTH, n_jobs=N_JOBS, 
                                       verbose=False)
        psd = results.get_data()
        freq = results.freqs
        
        # save power results
        fname_out = str.replace(fname, '_epo.fif', '_%s_psd' %(label))
        np.savez(join(dir_output, fname_out), psd=psd, freq=freq)


def compute_channel_tfr(epochs, fname, dir_output):
    '''
    This function takes an MNE epochsArray and computes the time-frequency
    representatoin of power for each channel sequentially, saving the results
    for each channel seperately. Data is downsampled to N_TFR_SAMPLES points.
    '''
    
    # get single channel epochs, and compute TFR for each channel
    for channel in range(len(epochs.info['ch_names'])):        
        # run time-frequency analysis
        decim = int(np.ceil(len(epochs.times) / N_TFR_SAMPLES))
        time, freq, tfr = compute_tfr(epochs, picks=channel, decim=decim)
        
        # save time-frequency results
        fname_out = fname.replace('_epo.fif', f'_chan{channel}_tfr')
        np.savez(join(dir_output, fname_out), 
                    tfr=tfr, freq=freq, time=time)

def compute_tfr(epochs, f_min=None, f_max=None, n_freqs=256, 
                time_window_length=0.5, freq_bandwidth=4, n_jobs=-1, picks=None, 
                average=False, decim=1, verbose=False):
    '''
    This function takes an MNE epochsArray and computes the time-frequency
    representatoin of power using the multitaper method. 
    Due to memory demands, this function should be run on single-channel data, 
    or results can be averaged across trials.
    '''
    
    # set paramters for TF decomposition
    if f_min is None:
        f_min = (1/(epochs.tmax-epochs.tmin)) # 1/T
    if f_max is None:
        f_max = epochs.info['sfreq'] / 2 # Nyquist

    freq = np.logspace(*np.log10([f_min, f_max]), n_freqs) # log-spaced 
    n_cycles = freq * time_window_length # set based on fixed window length
    time_bandwidth =  time_window_length * freq_bandwidth # must be >= 2

    # TF decomposition using multitapers
    tfr = tfr_multitaper(epochs, freqs=freq, n_cycles=n_cycles, 
                            time_bandwidth=time_bandwidth, return_itc=False, 
                            n_jobs=n_jobs, picks=picks, average=average, 
                            decim=decim, verbose=verbose)
    
    # extract data
    time = tfr.times
    tfr = tfr.data.squeeze()

    return time, freq, tfr


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
    conditions = ['words_hit_prestim', 'words_hit_poststim', 
                  'faces_hit_prestim', 'faces_hit_poststim',
                  'words_miss_prestim', 'words_miss_poststim', 
                  'faces_miss_prestim', 'faces_miss_poststim']
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
    
    # load channel meta data
    meta = np.load(join(PROJECT_PATH, 'data/ieeg_metadata', 
                        'ieeg_channel_info.pkl'), allow_pickle=True)
    
    # aggregate psd data for each condition
    for condition in ['words_hit', 'faces_hit', 'words_miss', 'faces_miss']:
        # display progress
        print(f'Aggregating condition: {condition}')

        # initialize dictionary for time-averaged tfr
        tfr_mean_pre = np.zeros([len(meta), len(freq)])
        tfr_mean_post = np.zeros([len(meta), len(freq)])
        tfr_mean_epoch = np.zeros([len(meta), len(freq)])
        
        # loop through rows of metadata
        for ii in range(len(meta)):
            # display progress 
            if ii % 100 == 0: 
                print(f"     files loaded: \t{ii} / {len(meta)}")

            # set file name
            fname_in = f"{meta['patient'][ii]}_{condition}_\
                chan{meta['chan_idx'][ii]}_tfr.npz"
            
            # skip missing channels
            if not fname_in in files: continue
        
            # load tfr data
            data_in = np.load(join(dir_input, fname_in))

            # if all values are NaN, set means to Nan and continue
            if np.all(np.isnan(data_in['tfr'])):
                tfr_mean_pre[ii] = np.nan
                tfr_mean_post[ii] = np.nan
                tfr_mean_epoch[ii] = np.nan
                continue

            # average across trials
            tfr = np.nanmedian(data_in['tfr'], axis=0)

            # crop time windows of interest
            tfr_eopch, _ = crop_tfr(tfr, data_in['time'], EPOCH_TIMES[0])
            tfr_pre, _ = crop_tfr(tfr, data_in['time'], EPOCH_TIMES[1])
            tfr_post, _ = crop_tfr(tfr, data_in['time'], EPOCH_TIMES[2])

            # average across time for each time window of interest
            tfr_mean_epoch[ii] = np.nanmean(tfr_eopch, axis=1)
            tfr_mean_pre[ii] = np.nanmean(tfr_pre, axis=1)
            tfr_mean_post[ii] = np.nanmean(tfr_post, axis=1)

        #  save results
        np.savez(join(dir_output, f"tfr_{condition}_epoch"), freq=freq, 
                 spectra=tfr_mean_epoch)
        np.savez(join(dir_output, f"tfr_{condition}_prestim"), freq=freq, 
                 spectra=tfr_mean_pre)
        np.savez(join(dir_output, f"tfr_{condition}_poststim"), freq=freq, 
                 spectra=tfr_mean_post)

if __name__ == "__main__":
    main()
    
    