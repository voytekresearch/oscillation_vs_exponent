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
from scipy.optimize import  minimize, Bounds
from scipy.stats import mode

from fooof import FOOOFGroup
from fooof.sim import gen_power_spectrum
from neurodsp.spectral import rotate_powerlaw

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
                    comp_intersection_from_params(param_pre, param_post)
                
                # # calc intersection using optimization
                # psd_pre, psd_post, intersection = solve_for_rotation_freq(N_ITER, param_pre, param_post, average=AVERAGE_METHOD)
        
                # save results
                fname_out = 'intersection_results_%s_%s_%s' %(material, memory, ap_mode)
                np.savez(join(dir_output, fname_out), psd_pre=psd_pre, psd_post=psd_post, 
                        intersection=intersection)
        

def comp_intersection_from_params(param_pre, param_post):
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

def rotate_psd(f_rotation, freqs, psd_0, psd_1, delta_exp):
    """
    Rotate power spectrum and compute the difference betweeen the rotated spectrum and a 
    second spectrum. This function is used to minimize the difference between the rotated
    and second spectrum. Spectra are log-transformed before teh difference is computed.
    The error between the two spectra is squared.

    Parameters
    ----------
    f_rotation : float
        frequency at which to rotate the power spectrum
    freqs : 1d array
        frequency vector
    psd_0 : 1d array
        power spectrum to rotate
    psd_1 : 1d array
        power spectrum to compare rotated spectrum to
    delta_exp : float
        degree of rotation (change in exponent)

    Returns
    -------
    diff : float
        difference between rotated and second spectrum (squared error)

    """

    # rotate
    psd_rot = rotate_powerlaw(freqs, psd_0, delta_exponent=delta_exp, f_rotation=f_rotation)
    
    # calc difference in power between parameter sets
    diff = np.sum((np.log10(psd_1) - np.log10(psd_rot))**2)
    
    return diff

def minimize_rotation_error(param_0, param_1, x0):
    # cont number of spectra
    n_spectra = len(param_0.get_params('r_squared'))
    
    # calc change/difference in exponent
    delta_exp = param_1.get_params('aperiodic', 'exponent') - param_0.get_params('aperiodic', 'exponent')
    
    # initialize array for results
    psd_0 = np.zeros([n_spectra, len(param_0.freqs)])
    psd_1 = psd_0.copy()
    intersection = np.zeros(n_spectra)
    intersection[:] = np.nan
    # loop through spectra
    for ii in range(n_spectra):
        # generate aperiodic spectra
        psd_freq, psd_0[ii] = gen_power_spectrum(param_0.freq_range, param_0.get_params('aperiodic')[ii], [], freq_res=param_0.freq_res)
        _, psd_1[ii] = gen_power_spectrum(param_1.freq_range, param_1.get_params('aperiodic')[ii], [], freq_res=param_1.freq_res)
        
        # find intersection
        result = minimize(rotate_psd, x0=x0, bounds=Bounds(lb=[param_0.freqs.min(),], ub=[param_0.freqs.max(),]),
                          args=(psd_freq, psd_0[ii], psd_1[ii], delta_exp[ii]))
        intersection[ii] = result['x']

    return psd_0, psd_1, intersection

def solve_for_rotation_freq(n_iter, param_0, param_1, average='mode'):
    
    # initialize output array
    intersection_rand = np.zeros([n_iter, len(param_0)])
    
    # solve for intersection
    for i_iter in range(N_ITER):
        x0 = param_0.freqs[np.random.randint(0, len(param_0.freqs))]
        psd_0, psd_1, intersection_rand[i_iter] = minimize_rotation_error(param_0, param_1, x0=[x0,])
    
    # take average over iterations
    if average == 'median':
        f_rot = np.nanmedian(intersection_rand, 0)
    elif average == 'mean':
        f_rot = np.nanmean(intersection_rand, 0)
    elif average == 'mode':
        # round solutions to nearest whole number
        # final solution must have a count > n_iter / 2
        f_rot, count = mode(np.round_(intersection_rand), 0)
        f_rot[count < n_iter/2] = np.nan
    
    return psd_0, psd_1, f_rot
    
if __name__ == "__main__":
    main()










