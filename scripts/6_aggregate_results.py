"""
This script aggregates the results from the spectral parameterization and 
intersection analyses (scripts 4 and 5) and saves them to a single dataframe.

"""

# SET PATH
PROJECT_PATH = 'C:/Users/micha/projects/oscillation_vs_exponent/'

# Imports - standard
import os
import numpy as np
import pandas as pd

# Imports - specparam
from fooof import FOOOFGroup
from fooof.analysis import get_band_peak_fg
from fooof.utils import trim_spectrum

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from info import ALPHA_RANGE
from specparam_utils import load_ap_params, params_to_spectra, compute_adj_r2

# Settings - spectral analysis hyperparameters
AP_MODE = 'knee' # aperiodic mode
DECOMP_METHOD = 'tfr' # analyze PSDs or average TFRs

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
            for epoch in ['pre', 'post']:
                # add condition info (material, memory, epoch)
                df = chan_info.copy()
                df['material'] = material
                df['memory'] = memory
                df['epoch'] = epoch

                # load aperiodic parameters
                fname_in = f"{DECOMP_METHOD}_{material}_{memory}_{epoch}_params_{AP_MODE}"
                print(f"Loading {fname_in}...")
                df['offset'], df['knee'], df['exponent'] = load_ap_params(f"{PROJECT_PATH}/data/ieeg_psd_param/{fname_in}")

                # load intersection frequency results
                fname_in = f"intersection_results_{material}_{memory}_{AP_MODE}.npz"
                data_in = np.load(f"{PROJECT_PATH}/data/ieeg_intersection_results/{fname_in}", allow_pickle=True)
                df['f_rotation'] = data_in['intersection']

                # load specparam results
                params = FOOOFGroup()
                fname_in = f"{DECOMP_METHOD}_{material}_{memory}_{epoch}_params_{AP_MODE}.json"
                params.load(f"{PROJECT_PATH}/data/ieeg_psd_param/{fname_in}")

                # add alpha results
                alpha = get_band_peak_fg(params, ALPHA_RANGE)
                df["alpha_cf"] = alpha[:,0]
                df["alpha_pw"] = alpha[:,1]
                df["alpha_bw"] = alpha[:,2]

                # add alpha power results
                data_in = np.load(f"{PROJECT_PATH}/data/ieeg_spectral_results/{DECOMP_METHOD}_{material}_{memory}_{epoch}stim.npz")
                _, alpha = trim_spectrum(data_in['freq'], data_in['spectra'], f_range=ALPHA_RANGE)
                df['alpha'] = np.nanmean(alpha, axis=1)

                # drop first index (0 Hz)
                spectra = data_in['spectra'][:,1:]
                freq = data_in['freq'][1:]

                # add adjusted alpha power results
                params.freqs = freq
                spectra_ap = params_to_spectra(params, component='aperiodic')
                spectra_adjusted = spectra - spectra_ap
                _, alpha_adj = trim_spectrum(freq,spectra_adjusted, f_range=ALPHA_RANGE)
                df['alpha_adj'] = np.nanmean(alpha_adj, axis=1)

                # add r-squared and adjusted r-squared
                df['r2'] = params.get_params('r_squared')
                df['r2_adj'] = compute_adj_r2(params)

                # add to list
                df_list.append(df)

    # join dataframes for hit and miss conditions
    df = pd.concat(df_list, ignore_index=True)
        
    # save dataframe for OLS analysis
    df.to_csv(f"{dir_output}/spectral_parameters.csv")


if __name__ == "__main__":
    main()