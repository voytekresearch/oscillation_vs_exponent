"""
This script aggregates the results from the spectral parameterization and 
intersection analyses (scripts 4 and 5) and saves them to a single dataframe.

"""


# Imports - standard
import os
import numpy as np
import pandas as pd

# Imports - specparam
from specparam import SpectralGroupModel
from specparam.bands import Bands

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from settings import AP_MODE, BANDS
from specparam_utils import (compute_adj_r2, compute_band_power, 
                             compute_adjusted_band_power)

# settings
BAND_POWER_METHOD = 'mean'
LOG_POWER = True

def main():
    # id directories
    dir_output = f"{PROJECT_PATH}/data/results"
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    # load channel info
    fname_in = f"{PROJECT_PATH}/data/ieeg_metadata/ieeg_channel_info.csv"
    chan_info = pd.read_csv(fname_in, index_col=0)

    # get data for each parameter and condition
    df_list = []
    for material in ['words', 'faces']:
        for memory in ['hit', 'miss']:
            for epoch in ['pre', 'post']:
                df = aggregate_results(chan_info, material, memory, epoch)
                df_list.append(df)

    # join dataframes for hit and miss conditions
    df = pd.concat(df_list, ignore_index=True)
        
    # save dataframe for OLS analysis
    df.to_csv(f"{dir_output}/spectral_parameters.csv")


def aggregate_results(chan_info, material, memory, epoch):
    # add channel and condition (material, memory, epoch)
    df = chan_info.copy()
    df['material'] = material
    df['memory'] = memory
    df['epoch'] = epoch

    # add specparam results
    sp = SpectralGroupModel()
    fname_in = f"psd_{material}_{memory}_{epoch}stim_params_{AP_MODE}"
    sp.load(f"{PROJECT_PATH}/data/ieeg_psd_param/{fname_in}")
    params = sp.to_df(Bands(BANDS))
    df = pd.concat([df, params], axis=1)

    # add adjusted r-squared
    df['r2_adj'] = compute_adj_r2(sp)

    # add intersection frequency results
    fname = f"intersection_results_{material}_{memory}_{AP_MODE}.npz"
    data_in = np.load(f"{PROJECT_PATH}/data/ieeg_intersection_results/{fname}", 
                        allow_pickle=True)
    df['f_rotation'] = data_in['intersection']

    # add band power results
    fname = f"psd_{material}_{memory}_{epoch}stim.npz"
    data_in = np.load(f"{PROJECT_PATH}/data/ieeg_spectral_results/{fname}")
    for band in BANDS:
        # add band power 
        power = compute_band_power(data_in['freq'], data_in['spectra'],
                                    BANDS[band], method=BAND_POWER_METHOD,
                                    log_power=LOG_POWER)
        df[band] = power

        # add adjusted band power
        power = compute_adjusted_band_power(sp, BANDS[band], 
                                            method=BAND_POWER_METHOD,
                                            log_power=LOG_POWER)
        df[f"{band}_adj"] = power


    return df


if __name__ == "__main__":
    main()
