"""
This script computes the intersaection frequency between the pre- and 
post-stimulus power spectra.

"""


# Imports - general
from os.path import join, exists
from os import mkdir
import numpy as np
from fooof import FOOOFGroup

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from specparam_utils import comp_intersection

# Settings
DECOMP_METHOD = 'tfr' # analyze PSDs or average TFRs

def main():
    # identify / create directories
    dir_input = join(PROJECT_PATH, 'data/ieeg_psd_param')
    dir_output = join(PROJECT_PATH, 'data/ieeg_intersection_results')
    if not exists(dir_output): 
        mkdir(dir_output)
        
    # loop through conditions
    for material in ['words', 'faces']:
        for memory in ['hit', 'miss']:
            for ap_mode in ['knee']: # ['fixed', 'knee']:
                # load parameterization results
                param_pre = FOOOFGroup()
                fname = f"{dir_input}/{DECOMP_METHOD}_{material}_{memory}_prestim_params_{ap_mode}.json"
                param_pre.load(fname)

                param_post = FOOOFGroup()
                fname = f"{dir_input}/{DECOMP_METHOD}_{material}_{memory}_poststim_params_{ap_mode}.json"
                param_post.load(fname)
        
        
                # calc intersection 
                temp = comp_intersection(param_pre, param_post)
                psd_pre, psd_post, intersection, intersection_idx = temp
                
                # save results
                fname_out = f'{dir_output}/intersection_results_{material}_{memory}_{ap_mode}.npz'
                np.savez(fname_out, psd_pre=psd_pre, psd_post=psd_post, 
                        intersection=intersection, 
                        intersection_idx=intersection_idx)

    
if __name__ == "__main__":
    main()










