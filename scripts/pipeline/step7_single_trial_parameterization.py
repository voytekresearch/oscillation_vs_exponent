"""
This script computes the difference in spectral parameters between pre- and
post-stimulus spectra for each trial in the dataset. The results are saved to a
CSV file for further analysis at .../data/results/single_trial_params.csv.
"""

# Imports - standard
import os
import numpy as np
import pandas as pd
from specparam import SpectralGroupModel, fit_models_3d
from specparam.utils import trim_spectrum

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from utils import get_start_time, print_time_elapsed
from info import PATIENTS, MATERIALS
from settings import SPEC_PARAM_SETTINGS, N_JOBS, AP_MODE


def main():

    # display progress
    t_start = get_start_time()

    # identify / create directories
    dir_output = f"{PROJECT_PATH}/data/results"
    if not os.path.exists(dir_output): 
        os.makedirs(f"{dir_output}")

    # init
    df = pd.DataFrame(columns=['patient', 'material', 'trial', 'channel', 
                                'behavior', 'exp_diff', 'alpha_diff', 
                                'gamma_diff'])
    
    # loop through patients, materials, and behaviors
    for patient in PATIENTS:
        for material in MATERIALS:
            for behavior in ['hit', 'miss']:
                # display progress
                print(f"Processing {patient}, {material}, {behavior}...")

                # load data
                data_pre = np.load(f"{PROJECT_PATH}/data/ieeg_psd/"+\
                    f"{patient}_{material}_{behavior}_prestim_psd.npz")
                data_post = np.load(f"{PROJECT_PATH}/data/ieeg_psd/"+\
                    f"{patient}_{material}_{behavior}_poststim_psd.npz")

                # run analysis
                results = run_analysis(data_pre['psd'], data_post['psd'], 
                                       data_pre['freq']) 

                # add to dataframe
                for i_trial in range(results[0].shape[0]):
                    for i_chan in range(results[0].shape[1]):
                        results_ii = pd.DataFrame({
                            'patient'   :   patient,
                            'material'  :   material,
                            'trial'     :   i_trial,
                            'channel'   :   i_chan,
                            'behavior'  :   behavior,
                            'exp_diff'  :   results[0][i_trial, i_chan],
                            'alpha_diff':   results[1][i_trial, i_chan],
                            'gamma_diff':   results[2][i_trial, i_chan]}, 
                            index=[0])
                        
                        df = pd.concat([df, results_ii], axis=0)

    # save results
    df.to_csv(f"{dir_output}/single_trial_params.csv", index=False)

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)

    
def run_analysis(spectra_pre, spectra_post, freq):
    """
    Run analysis on example data. Apply spectral parameterization and compute
    parameter differences between pre- and post-stimulus spectra.

    Parameters
    ----------
    spectra_pre, spectra_post : 2d array
        Spectra to compute parameter differences between.
    freq : 1d array
        Frequency values for spectra.

    Returns
    -------
    exp_diff, alpha_diff, gamma_diff : 2d array
        Parameter differences between pre- and post-stimulus spectra.
    """

    # apply SpecParam
    sgm = SpectralGroupModel(**SPEC_PARAM_SETTINGS, aperiodic_mode=AP_MODE, 
                             verbose=False)
    sgm.set_check_modes(check_data=False)
    sgm_pre, sgm_post = sgm.copy(), sgm.copy()
    fit_models_3d(sgm_pre, freq, spectra_pre, n_jobs=N_JOBS)
    fit_models_3d(sgm_post, freq, spectra_post, n_jobs=N_JOBS)

    # compute parameter differences for each SpectralModel
    exp_diff = np.zeros(len(sgm_pre))
    alpha_diff = np.zeros(len(sgm_pre))
    gamma_diff = np.zeros(len(sgm_pre))
    for idx in range(len(sgm_pre)):
        results = compute_parameter_diff(sgm_pre.get_model(idx), 
                                         sgm_post.get_model(idx))
        exp_diff[idx], alpha_diff[idx], gamma_diff[idx] = results
        
    # reshape results to match original data shape
    exp_diff = exp_diff.reshape(spectra_pre.shape[:2])
    alpha_diff = alpha_diff.reshape(spectra_pre.shape[:2])
    gamma_diff = gamma_diff.reshape(spectra_pre.shape[:2])

    return exp_diff, alpha_diff, gamma_diff
    

def compute_parameter_diff(sm_0, sm_1, alpha_band=[8, 20], gamma_band=[50, 90]):
    """
    Compute difference in spectral statistics between two spectral models, 
    including the aperiodic exponent, alpha band power, and gamma band power.

    Parameters
    ----------
    sm_0, sm_1 : SpectralModel
        Spectral models to compute parameter difference between.
    alpha_band, gamma_band : list of float
        Frequency ranges to compute band power over.

    Returns
    -------
    exp_diff, alpha_diff, gamma_diff : float
        Difference in spectral parameters between the two spectral models.
    
    """

    # compute difference in aperiodic exponent
    exp_diff = sm_1.aperiodic_params_[2] - sm_0.aperiodic_params_[2]

    # compute difference in alpha band power
    alpha_0 = compute_band_power(sm_0, alpha_band)
    alpha_1 = compute_band_power(sm_1, alpha_band)
    alpha_diff = alpha_1 - alpha_0

    # compute difference in gamma band power
    gamma_0 = compute_band_power(sm_0, gamma_band)
    gamma_1 = compute_band_power(sm_1, gamma_band)
    gamma_diff = gamma_1 - gamma_0

    return exp_diff, alpha_diff, gamma_diff


def compute_band_power(sm, f_range):
    """
    Compute band power for a given spectral model.

    Parameters
    ----------
    sm : SpectralModel
        Spectral model to compute band power for.
    f_range : list of float
        Frequency range to compute band power over.

    Returns
    -------
    power : float
        Band power for the given spectral model.
    
    """

    # compute flattened spectrum
    spectrum_flat = sm.power_spectrum - sm._ap_fit

    # compute band power
    _, band_power = trim_spectrum(sm.freqs, spectrum_flat, f_range)

    # take integral
    power = band_power.sum()

    return power


if __name__ == "__main__":
    main()
