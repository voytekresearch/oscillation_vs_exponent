# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 17:29:53 2021

@author: micha

Data Repo: https://osf.io/3csku/
Associated Paper: https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000403

The provided dataset is formatted as Fieldtrip data structures (.mat);  
this script reformats the iEEG dataset into MNE epochsArrays (.fif). The 
time-series are also saved as numpy arrays (.npy).

"""

# Imports - standard
from os.path import join, exists
from os import makedirs
import mne
from pymatreader import read_mat
import numpy as np
import pandas as pd

# Imports - custom
import sys
sys.path.append("code")
from info import PATIENTS, FS
from paths import PROJECT_PATH, DATASET_PATH


def main():
    """
    convert Fieldtrip data structures to MNE epochs objects
    additionally, aggregate info about the dataset
    
    """
        
    # create dataframe for metadata
    columns = ['patient', 'material', 'chan_idx', 'label', 'pos_y', 'pos_x', 
               'pos_z']
    meta = pd.DataFrame(columns=columns)
    
    # loop through all files in dataset
    dir_input = join(DATASET_PATH, 'iEEG')
    for patient in PATIENTS:
        for material in ['word', 'face']:
            fname = '%s_%ss.mat' %(patient, material)
            
            # display progress
            print('\n__________Reformatting: %s ____________________\n' %fname)
        
            # import epoch data
            epochs = import_epochs(join(dir_input, fname))
    
            # export epochs data
            save_epochs(epochs, fname)
    
            # collect channel info for file
            info = collect_channel_info(dir_input, fname)
            meta = pd.concat([meta, info], sort=False)
        
    # export aggregate channel info
    save_metadata(meta, join(PROJECT_PATH, 'data/ieeg_metadata'))

def create_montage(fname):
    """
    create digital channel montage for MNE epochs array
    
    """
    
    # load fieldtrip data structure
    data_in = read_mat(fname)

    # get channel info from data sructure
    label = data_in['data']['elecinfo']['label_bipolar']
    elecpos = data_in['data']['elecinfo']['elecpos_bipolar']
    
    # create montage
    ch_pos = dict(zip(label, elecpos))
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
        
    return montage

def import_epochs(fname):
    """
    import Fieldtrip data structure as MNE epochs array
    
    """
    
    # create channel montage for epochsArray
    # note: channel names are needed to import Fieldtrip epochs using MNE
    montage = create_montage(fname)

    # import Fieldtrip data structure
    info = mne.create_info(montage.ch_names, FS, ch_types='eeg')
    epochs = mne.read_epochs_fieldtrip(fname, info)

    # # set montage
    epochs.set_montage(montage)
    
    # label metadata
    epochs.metadata.rename(columns={0:'trial_num', 1:'pleasantness', 
                                    2:'confidence', 3:'recalled', 
                                    4:'reaction_time'}, inplace=True)
    
    return epochs

def save_epochs(epochs, fname):
    """
    export MNE epochs array - save as .fif and .npy
    save an additional .fif after removing unsuccessful trials.
    
    """
    
    # identify / create directories
    dir_dataset = join(PROJECT_PATH, 'data/ieeg_dataset')
    dir_output = join(PROJECT_PATH, 'data/ieeg_epochs')
    
    if not exists(dir_output): makedirs(dir_output)
    if not exists(join(dir_dataset, 'fif')): makedirs(join(dir_dataset, 'fif'))
    if not exists(join(dir_dataset, 'npy')): makedirs(join(dir_dataset, 'npy'))

    # save data as .fif 
    epochs.save(join(dir_dataset, 'fif', fname.replace('.mat','_epo.fif')),
                overwrite=True)

    # save data as .npy
    lfp = epochs.get_data()
    np.save(join(dir_dataset, 'npy', fname.replace('.mat', '.npy')), lfp)
    
    # split successful and unsuccessful trials
    epochs_hit = epochs[epochs.metadata['recalled'].values.astype('bool')]
    epochs_miss = epochs[~epochs.metadata['recalled'].values.astype('bool')]

    # save data as .fif - after dropping unsuccessful trials 
    epochs_hit.save(join(dir_output, fname.replace('.mat', '_hit_epo.fif')), overwrite=True)
    epochs_miss.save(join(dir_output, fname.replace('.mat', '_miss_epo.fif')), overwrite=True)
    
def collect_channel_info(dir_input, fname):
    """
    generate dataframe containing electrode info, including channel location
    
    """
    
    # create dataframe for channel info
    columns = ['patient', 'material', 'chan_idx', 'label', 'pos_x', 'pos_y',
               'pos_z']
    info = pd.DataFrame(columns=columns)
    
    # get channel locations from Fieldtrip data structure
    data_in = read_mat(join(dir_input, fname))
    label = data_in['data']['elecinfo']['label_bipolar']
    info['label'] = label
    elecpos = data_in['data']['elecinfo']['elecpos_bipolar']
    for ii in range(elecpos.shape[0]):
        info['pos_x'][ii], info['pos_y'][ii], info['pos_z'][ii] = elecpos[ii]
    
    # Get metadata from filename
    f_parts = fname.split('_')
    info['patient'] = [f_parts[0]] * len(label)
    info['material'] = [f_parts[1].replace('.mat', '')] * len(label)
    info['chan_idx'] = np.arange(len(label))
    
    return info

def save_metadata(meta, dir_output):
    # make folder for output
    if not exists(dir_output): makedirs(dir_output)
    
    # remove duplicates reset index
    meta = meta[meta['material'] == 'faces']
    meta = meta.drop(columns='material')
    meta.reset_index(inplace=True)
    
    # save metadata to file
    meta.to_pickle(join(dir_output, 'ieeg_channel_info.pkl'))
    meta.to_csv(join(dir_output, 'ieeg_channel_info.csv'))        

if __name__ == "__main__":
    main()
    
