"""
This script computes the ERP for each channel in the dataset, and computes
several ERP metrics. The ERP amplitude is computed as the ratio of the maximum 
ERP amplitude after the stimulus onset to the maximum ERP amplitude before the 
stimulus onset. The integral of the ERP for both windows is also computed.


"""

# Imports - general
import os
import numpy as np
import pandas as pd
from mne import read_epochs
from time import time as timer

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from utils import hour_min_sec
from tfr_utils import crop_tfr
from erp_utils import compute_erp

# settings 
T_BASELINE = [-0.5, 0.] # baseline time window for ERP computation
T_TRIM = [-0.5, 2.] # time window to trim signal    


def main():
    # display progress
    t_start = timer()

    # identify / create directories
    dir_input = f"{PROJECT_PATH}/data/ieeg_epochs"
    dir_output = f"{PROJECT_PATH}/data/results"
    dir_fig = f"{PROJECT_PATH}/figures/erp"
    if not os.path.exists(dir_output): os.makedirs(dir_output)
    if not os.path.exists(dir_fig): os.makedirs(dir_fig)
    
    # init
    df_list = []

    # for each fif file
    files = os.listdir(dir_input)
    for ii, fname in enumerate(files):

        # load eeg data
        epochs = read_epochs(f"{dir_input}/{fname}", verbose=False)
        signals = epochs.get_data(copy=True)
        time = epochs.times
        print(f"\tchannels: \t{len(epochs.info['ch_names'])}")
    
        # compute erp
        erps = compute_erp(signals, time, T_BASELINE)

        # trim signal
        erps, time = crop_tfr(erps, time, T_TRIM)

        # compute ERP amplitude and add to df
        erps_abs = np.abs(erps)
        erp_max_pre = np.nanmax(erps_abs[:, time<0], axis=1)
        erp_max_post = np.nanmax(erps_abs[:, time>0], axis=1)
        erp_max = erp_max_post / erp_max_pre
        erp_int_pre = np.sum(erps_abs[:, time<0], axis=1)
        erp_int_post = np.sum(erps_abs[:, time>0], axis=1)
        erp_int = erp_int_post / erp_int_pre
        fparts = fname.split('_')
        df_i = pd.DataFrame({'patient' : fparts[0],
                            'chan_idx' : np.arange(erps.shape[0]),
                            'material' : fparts[1],
                            'memory' : fparts[2],
                            'erp_max_pre' : erp_max_pre,
                            'erp_int_pre' : erp_int_pre,
                            'erp_max_post' : erp_max_post,
                            'erp_int_post' : erp_int_post,
                            'erp_max' : erp_max,
                            'erp_int' : erp_int})
        df_list.append(df_i)

    # save df
    df_erp = pd.concat(df_list)
    df_erp.to_csv(f"{dir_output}/df_erp.csv", index=False)

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\n\nTotal analysis time: {hour} hour, {min} min, and {sec :0.1f} s")
    

if __name__ == "__main__":
    main()
    
    