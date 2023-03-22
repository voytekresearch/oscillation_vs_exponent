# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 11:46:53 2021

@author: micha
"""

# SET PATH
PROJECT_PATH = 'C:/Users/micha/projects/oscillation_vs_exponent/'

# Imports - general
import os
import numpy as np
import pandas as pd

# Imports - specparam
from fooof import FOOOFGroup
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fg
from fooof.utils import trim_spectrum

# Imports - custom
from utils import load_ap_params, params_to_spectra

# Parameters - dataset details
FS = 512 # sampling frequency
TMIN = -1.5 # epoch start time
PATIENTS = ['pat02','pat04','pat05','pat08','pat10','pat11','pat15','pat16',
            'pat17','pat19','pat20','pat21','pat22'] # subject IDs

# Parameters - spectral analysis hyperparameters
bands = Bands({'alpha' : [7, 13]}) # define spectral bands of interest
AP_MODE = 'knee' # aperiodic mode

def main():
    # id directories
    dir_output = f"{PROJECT_PATH}/data/results"
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    # load channel info
    chan_info = pd.read_pickle(F"{PROJECT_PATH}/data/ieeg_metadata/ieeg_channel_info.pkl")
    chan_info.drop(columns=['index'], inplace=True)
    chan_info['unique_id'] = chan_info['patient'] + '_' + chan_info['chan_idx'].astype(str)

    # get data for each parameter and condition
    df_list = []
    for material in ['words', 'faces']:
        for memory in ['hit', 'miss']:
            for epoch in ['prestim', 'poststim']:
                # add condition info (material, memory, epoch)
                df = chan_info.copy()
                df['material'] = material
                df['memory'] = memory
                df['epoch'] = epoch

                # load aperiodic parameters
                fname_in = f"{material}_{memory}_{epoch}_params_{AP_MODE}"
                df['offset'], df['knee'], df['exponent'] = load_ap_params(f"{PROJECT_PATH}/data/ieeg_psd_param/{fname_in}")

                # load intersection frequency results
                fname_in = f"intersection_results_{material}_{memory}_{AP_MODE}.npz"
                data_in = np.load(f"{PROJECT_PATH}/data/ieeg_intersection_results/{fname_in}", allow_pickle=True)
                df['f_rotation'] = data_in['intersection']

                # load alpha results
                params = FOOOFGroup()
                fname_in = f"{material}_{memory}_{epoch}_params_{AP_MODE}.json"
                params.load(f"{PROJECT_PATH}/data/ieeg_psd_param/{fname_in}")
                alpha = get_band_peak_fg(params, bands['alpha'])
                df["alpha_cf"] = alpha[:,0]
                df["alpha_pw"] = alpha[:,1]
                df["alpha_bw"] = alpha[:,2]

                # add alpha bandpower results
                data_in = np.load(f"{PROJECT_PATH}/data/ieeg_spectral_results/psd_{material}_{memory}_{epoch}.npz")
                _, alpha_bp = trim_spectrum(data_in['freq'], data_in['spectra'], f_range=bands['alpha'])
                df['alpha_bp'] = np.nanmean(alpha_bp, axis=1)

                # add adjusted alpha bandpower results
                spectra_ap = params_to_spectra(params, component='aperiodic')
                spectra_adjusted = data_in['spectra'] - spectra_ap
                _, alpha_adj = trim_spectrum(data_in['freq'],spectra_adjusted, f_range=bands['alpha'])
                df['alpha_adj'] = np.nanmean(alpha_adj, axis=1)                

                # add to list
                df_list.append(df)

    # join dataframes for hit and miss conditions
    df = pd.concat(df_list, ignore_index=True)
        
    # save dataframe for OLS analysis
    df.to_csv(f"{dir_output}/spectral_parameters.csv")


if __name__ == "__main__":
    main()