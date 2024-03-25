"""
This script computes the intersection frequency of the baseline and encoding  
power spectra. It analyzes the spectral results from 
scripts.ieeg_4_spectral_parameterization.py

"""


# Imports - general
from os.path import join, exists
from os import mkdir
import numpy as np
from specparam import SpectralGroupModel

# import - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from specparam_utils import compute_intersections

# settings
AP_MODE = ['knee'] # array. aperiodic modes for SpecParam. 

def main():
    # identify / create directories
    dir_input = join(PROJECT_PATH, 'data/ieeg_psd_param')
    dir_output = join(PROJECT_PATH, 'data/ieeg_intersection_results')
    if not exists(dir_output): 
        mkdir(dir_output)
        
    # loop through conditions
    for material in ['words', 'faces']:
        for memory in ['hit', 'miss']:
            for ap_mode in AP_MODE:
                # load parameterization results
                param_pre = SpectralGroupModel()
                param_pre.load(join(dir_input, 'psd_%s_%s_prestim_params_%s.json' %(material, memory, ap_mode)))
                param_post = SpectralGroupModel()
                param_post.load(join(dir_input, 'psd_%s_%s_poststim_params_%s.json' %(material, memory, ap_mode)))
        
                # calc intersection 
                results = compute_intersections(param_pre, param_post)
                intersection, intersection_idx = results
                
                # save results
                fname_out = f"intersection_results_{material}_{memory}_{ap_mode}"
                np.savez(join(dir_output, fname_out), intersection=intersection, 
                         intersection_idx=intersection_idx)

    
if __name__ == "__main__":
    main()










