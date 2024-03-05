"""
This script reformats the iEEG dataset from Fieldtrip data structures (.mat) 
into MNE epochsArrays (.fif). The time-series are also saved as numpy arrays 
(.npy).

Data Repository: 
  https://osf.io/3csku/
  
Associated Paper: 
  https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000403

"""

# Imports - standard
import os
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
    dir_input = f"{DATASET_PATH}/iEEG"
    for patient in PATIENTS:
        for material in ['words', 'faces']:
            fname = f"{patient}_{material}.mat"
            
            # display progress
            print('\n__________Reformatting: %s ____________________\n' %fname)
        
            # import epoch data
            epochs = import_epochs(f"{dir_input}/{fname}")
    
            # export epochs data
            save_epochs(epochs, fname)
    
            # collect channel info for file
            info = collect_channel_info(dir_input, fname)
            meta = pd.concat([meta, info], sort=False, ignore_index=True)
        
    # export aggregate channel info
    save_metadata(meta, f"{PROJECT_PATH}/data/ieeg_metadata")


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
    dir_dataset = f'{PROJECT_PATH}/data/ieeg_dataset'
    dir_output = f'{PROJECT_PATH}/data/ieeg_epochs'
    for path in [dir_output, f"{dir_dataset}/fif", f"{dir_dataset}/npy"]:
        if not os.path.exists(path): 
            os.makedirs(f"{path}")

    # save data as .fif 
    epochs.save(f"{dir_dataset}/fif/{fname.replace('.mat','_epo.fif')}",
                overwrite=True)

    # save data as .npy
    lfp = epochs.get_data()
    np.save(f"{dir_dataset}/npy/{fname.replace('.mat', '.npy')}", lfp)
    
    # split successful and unsuccessful trials
    epochs_hit = epochs[epochs.metadata['recalled'].values.astype('bool')]
    epochs_miss = epochs[~epochs.metadata['recalled'].values.astype('bool')]

    # save epoch data for successful and unsuccessful trials
    epochs_hit.save(f"{dir_output}/{fname.replace('.mat', '_hit_epo.fif')}", 
                    overwrite=True)
    epochs_miss.save(f"{dir_output}/{fname.replace('.mat', '_miss_epo.fif')}", 
                     overwrite=True)
    
def collect_channel_info(dir_input, fname):
    """
    generate dataframe containing electrode info, including channel location
    
    """
    
    # create dataframe for channel info
    columns = ['patient', 'material', 'chan_idx', 'label', 'pos_x', 'pos_y',
               'pos_z']
    info = pd.DataFrame(columns=columns)
    
    # get channel locations from Fieldtrip data structure
    data_in = read_mat(f"{dir_input}/{fname}")
    label = data_in['data']['elecinfo']['label_bipolar']
    info['label'] = label
    elecpos = data_in['data']['elecinfo']['elecpos_bipolar']
    for ii in range(elecpos.shape[0]):
        info.loc[ii, ['pos_x', 'pos_y', 'pos_z']] = elecpos[ii]
    
    # Get metadata from filename
    f_parts = fname.split('_')
    info['patient'] = [f_parts[0]] * len(label)
    info['material'] = [f_parts[1].replace('.mat', '')] * len(label)
    info['chan_idx'] = np.arange(len(label))
    
    return info

def save_metadata(meta, dir_output):
    # make folder for output
    if not os.path.exists(dir_output): 
        os.makedirs(f"{dir_output}")
    
    # remove duplicates reset index
    meta = meta[meta['material'] == 'faces']
    meta = meta.drop(columns='material')
    meta.reset_index(inplace=True)
    
    # save metadata to file
    meta.to_csv(f"{dir_output}/ieeg_channel_info.csv")

if __name__ == "__main__":
    main()
    
