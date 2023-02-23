# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 09:27:01 2021

Data Repo: https://osf.io/3csku/
Associated Paper: https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000403

This script analyzes the spectral results from 
ieeg_3_spectral_parameterization.py

"""


# Imports

from os.path import join, exists
from os import mkdir
import numpy as np

from fooof import FOOOFGroup
from fooof.sim import gen_power_spectrum

# Parameters
PROJECT_PATH = 'C:/Users/micha/projects/oscillation_vs_exponent/'

# optimization settings
N_ITER = 10
AVERAGE_METHOD = 'mode'

def main():
    # identify / create directories
    dir_input = join(PROJECT_PATH, 'data/ieeg_psd_param')
    dir_output = join(PROJECT_PATH, 'data/ieeg_intersection_results')
    if not exists(dir_output): 
        mkdir(dir_output)
        
        
    # loop through conditions
    for material in ['words', 'faces']:
        for memory in ['hit', 'miss']:
            for ap_mode in ['fixed', 'knee']:
                # load parameterization results
                param_pre = FOOOFGroup()
                param_pre.load(join(dir_input, '%s_%s_prestim_params_%s.json' %(material, memory, ap_mode)))
                param_post = FOOOFGroup()
                param_post.load(join(dir_input, '%s_%s_poststim_params_%s.json' %(material, memory, ap_mode)))
        
        
                # calc intersection 
                psd_pre, psd_post, intersection, intersection_idx = \
                    comp_intersection(param_pre, param_post)
                
                # save results
                fname_out = 'intersection_results_%s_%s_%s' %(material, memory, ap_mode)
                np.savez(join(dir_output, fname_out), psd_pre=psd_pre, psd_post=psd_post, 
                        intersection=intersection)
        

def comp_intersection(param_pre, param_post):
    """ 
    Calculate intersection of pre and post stim psd

    Parameters
    ----------
    param_pre : FOOOFGroup
        FOOOFGroup object containing pre-stimulus parameters
    param_post : FOOOFGroup
        FOOOFGroup object containing post-stimulus parameters        

    Returns
    -------
    psd_pre : 1d array
        pre-stimulus spectra
    psd_post : 1d array
        post-stimulus spectra
    intersection : 1d array
        intersection frequency
    intersection_idx : 1d array
        index of intersection frequency

    """
    
    # count channels
    n_chans = len(param_pre.get_params('r_squared'))
    
    if n_chans == 1:
        # generate aperiodic spectra from parameters
        _, psd_pre = gen_power_spectrum(param_pre.f_range, param_pre.get_params('aperiodic'), [], freq_res=param_pre.freq_res)
        _, psd_post = gen_power_spectrum(param_post.f_range, param_post.get_params('aperiodic'), [], freq_res=param_pre.freq_res)

        # calc intersect of aperiodic spectra
        idx = np.argwhere(np.diff(np.sign(psd_post - psd_pre))).flatten()
        if idx.any(): 
            intersection = param_pre.freqs[np.squeeze(idx)]
            intersection_idx = np.squeeze(idx)
            
    elif n_chans > 1:
        # initialize variables
        psd_pre = np.zeros([n_chans, len(param_pre.freqs)])
        psd_post = psd_pre.copy()
        intersection = np.zeros([n_chans])
        intersection[:] = np.nan
        intersection_idx = intersection.copy()

        for chan in range(n_chans):
            # generate aperiodic spectra from parameters
            _, psd_pre[chan] = gen_power_spectrum(param_pre.freq_range, param_pre.get_params('aperiodic')[chan], [], freq_res=param_pre.freq_res, nlv=0)
            _, psd_post[chan] = gen_power_spectrum(param_post.freq_range, param_post.get_params('aperiodic')[chan], [], freq_res=param_post.freq_res, nlv=0)
            
            # calc intersect of aperiodic spectra
            idx = np.argwhere(np.diff(np.sign(psd_post[chan] - psd_pre[chan]))).flatten()

            # if no intersect or multiple intersects 
            if not idx.any(): 
                continue
            elif len(idx)==1: 
                intersection[chan] = param_pre.freqs[np.squeeze(idx)]
                intersection_idx[chan] = np.squeeze(idx)
            elif len(idx)==2: 
                intersection[chan] = param_pre.freqs[np.max(idx)]
                intersection_idx[chan] = np.max(idx)
            elif len(idx)==len(param_pre.freqs):
                intersection[chan] = np.nan
                intersection_idx[chan] = np.nan
                
    else:
        intersection = np.nan
        intersection_idx = np.nan
        print('check size of input')
        
    return psd_pre, psd_post, intersection, intersection_idx

    
if __name__ == "__main__":
    main()










