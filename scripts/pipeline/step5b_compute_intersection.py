"""
This script computes the intersection frequency of the baseline and encoding  
power spectra. It analyzes the spectral results from 
scripts.ieeg_7_single_trial_parameterization.py

"""


# Imports - general
import os
import pandas as pd
from specparam import SpectralGroupModel

# import - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from info import PATIENTS
from specparam_utils import compute_intersections

# settings
AP_MODE = ['knee'] # array. aperiodic modes for SpecParam. 

def main():
    # identify / create directories
    dir_input = f"{PROJECT_PATH}/data/ieeg_psd_trial_params"
    dir_output = f"{PROJECT_PATH}/data/results"
    if not os.path.exists(dir_output): 
        os.makedirs(dir_output)
        
    df_list = []
    # loop through all patients and conditions
    for patient in PATIENTS:
        for material in ['words', 'faces']:
            for memory in ['hit', 'miss']:
                for ap_mode in AP_MODE:
                    i_trial = 0
                    while True:
                        try:
                            # load parameterization results
                            param_pre = SpectralGroupModel()
                            param_pre.load(f"{dir_input}/{patient}_{material}_{memory}_prestim_psd_params_{ap_mode}_{i_trial}.json")
                            param_post = SpectralGroupModel()
                            param_post.load(f"{dir_input}/{patient}_{material}_{memory}_poststim_psd_params_{ap_mode}_{i_trial}.json")
                        except FileNotFoundError:
                            print(f"File not found: {dir_input}/{patient}_{material}_{memory}_poststim_psd_params_{ap_mode}_{i_trial}.json")
                            break

                        # calc intersection 
                        results = compute_intersections(param_pre, param_post)
                        
                        # store results
                        df_list.append({'patient': patient,
                                        'material': material,
                                        'memory': memory, 
                                        'ap_mode': ap_mode,
                                        'i_trial': i_trial,
                                        'intersection': results[0],
                                        'intersection_idx': results[1]})
                        
                        # iterate trial
                        i_trial += 1
                    
    # save results
    df = pd.DataFrame(df_list)
    df.to_csv(os.path.join(dir_output, 'trial_intersection_results.csv'), index=False)
    
if __name__ == "__main__":
    main()










