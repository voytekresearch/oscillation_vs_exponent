# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 09:27:01 2021

Data Repo: https://osf.io/3csku/
Associated Paper: https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000403

This script analyzes the spectral results from 
ieeg_3_spectral_parameterization.py

"""


# Imports - general
from os.path import join, exists
from os import mkdir
import numpy as np

from fooof import FOOOFGroup

# import - custom
from specparam_utils import comp_intersection

# Parameters
PROJECT_PATH = 'C:/Users/micha/projects/oscillation_vs_exponent/'

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
                param_pre.load(join(dir_input, 'psd_%s_%s_prestim_params_%s.json' %(material, memory, ap_mode)))
                param_post = FOOOFGroup()
                param_post.load(join(dir_input, 'psd_%s_%s_poststim_params_%s.json' %(material, memory, ap_mode)))
        
        
                # calc intersection 
                psd_pre, psd_post, intersection, intersection_idx = \
                    comp_intersection(param_pre, param_post)
                
                # save results
                fname_out = 'intersection_results_%s_%s_%s' %(material, memory, ap_mode)
                np.savez(join(dir_output, fname_out), psd_pre=psd_pre, psd_post=psd_post, 
                        intersection=intersection, intersection_idx=intersection_idx)

    
if __name__ == "__main__":
    main()










