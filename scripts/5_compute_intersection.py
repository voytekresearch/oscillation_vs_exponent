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
                param_pre.load(join(dir_input, '%s_%s_%s_prestim_params_%s.json' %(DECOMP_METHOD, material, memory, ap_mode)))
                param_post = FOOOFGroup()
                param_post.load(join(dir_input, '%s_%s_%s_poststim_params_%s.json' %(DECOMP_METHOD, material, memory, ap_mode)))
        
        
                # calc intersection 
                psd_pre, psd_post, intersection, intersection_idx = \
                    comp_intersection(param_pre, param_post)
                
                # save results
                fname_out = 'intersection_results_%s_%s_%s' %(material, memory, ap_mode)
                np.savez(join(dir_output, fname_out), psd_pre=psd_pre, psd_post=psd_post, 
                        intersection=intersection, intersection_idx=intersection_idx)

    
if __name__ == "__main__":
    main()










