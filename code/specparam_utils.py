# -*- coding: utf-8 -*-
"""
Utility functions for working with specparam results/objects
"""

# Imports
import numpy as np


def knee_freq(knee, exponent):
    """
    Convert specparam knee parameter to Hz.

    Parameters
    ----------
    knee, exponent : 1D array
        Knee and exponent parameters from specparam.

    Returns
    -------
    knee_hz : 1D array
        Knee in Hz.
    """
    knee_hz = np.zeros_like(knee)
    for ii in range(len(knee)):
        knee_hz[ii] = knee[ii]**(1/exponent[ii])
    
    return knee_hz


def extract_ap_params(params):
    """
    Extract aperiodic parameters from FOOOFGroup object.

    Parameters
    ----------
    params : FOOOFGroup object
        FOOOFGroup object containing aperiodic parameters.

    Returns
    -------
    offset, knee, exponent : 1D array
        Offset, knee, and exponent parameters.
    
    """
    # get aperiodic parameter time-series
    offset = params.get_params('aperiodic', 'offset')
    exponent = params.get_params('aperiodic', 'exponent')
    
    # check if knee data exists
    try:
        # convert k to Hz
        k = params.get_params('aperiodic', 'knee')
        knee = knee_freq(k, exponent)
    except:
        knee = [np.nan] * len(offset)
    
    return offset, knee, exponent


def load_ap_params(fname):
    """
    Load specparam results from file and extract aperiodic parameters.

    Parameters
    ----------
    fname : str
        Filename to load.

    Returns
    -------
    offset, knee, exponent : 1D array
        Offset, knee, and exponent parameters.
    """

    # imports
    from fooof import FOOOFGroup

    # import specparam results
    params = FOOOFGroup()
    params.load(fname)
    
    return extract_ap_params(params)


def params_to_spectra(params, component='both'):
    """
    Simulate aperiodic power spectra from FOOOFGroup object.

    Parameters
    ----------
    params : FOOOFGroup object
        FOOOFGroup object containing aperiodic parameters.
    component : str
        Component to simulate ('both', 'aperiodic', or 'peak'). Default is 'both'.

    Returns
    -------
    spectra: array
        Aperiodic power spectra.

    """
    # imports
    from fooof.sim import gen_power_spectrum

    # init
    spectra = np.zeros([len(params), len(params.freqs)])
    
    # simulate aperiodic spectra for each
    if component == 'both':
        for ii in range(len(params)):
            _, spectra[ii] = gen_power_spectrum([params.freqs[0], params.freqs[-1]],
                                                params[ii].aperiodic_params, params[ii].peak_params, 
                                                freq_res=params.freq_res, nlv=0)
    elif component == 'aperiodic':
        for ii in range(len(params)):
            _, spectra[ii] = gen_power_spectrum([params.freqs[0], params.freqs[-1]],
                                                params[ii].aperiodic_params, [], 
                                                freq_res=params.freq_res, nlv=0)
    # simulate aperiodic spectra for each
    elif component == 'peak':
        for ii in range(len(params)):
            _, spectra[ii] = gen_power_spectrum([params.freqs[0], params.freqs[-1]],
                                                [], params[ii].peak_params, 
                                                freq_res=params.freq_res, nlv=0)
    else:
        raise ValueError('Invalid component specified. Must be "both", "aperiodic", or "peak".')
    
    return spectra

def params_to_spectrum(params, component='both'):
    """
    Simulate aperiodic power spectra from FOOOFGroup object.

    Parameters
    ----------
    params : FOOOF object
        FOOOF object containing aperiodic parameters.
    component : str
        Component to simulate ('both', 'aperiodic', or 'peak'). Default is 'both'.

    Returns
    -------
    spectra: array
        Aperiodic power spectra.

    """
    # imports
    from fooof.sim import gen_power_spectrum

    # simulate aperiodic spectra for each
    if component == 'both':
        _, spectrum = gen_power_spectrum([params.freqs[0], params.freqs[-1]],
                                            params.get_params('aperiodic'),
                                            params.get_params('peak'),
                                            freq_res=params.freq_res, nlv=0)
    elif component == 'aperiodic':
        _, spectrum = gen_power_spectrum([params.freqs[0], params.freqs[-1]],
                                            params.get_params('aperiodic'), [], 
                                            freq_res=params.freq_res, nlv=0)
    # simulate aperiodic spectra for each
    elif component == 'peak':
        _, spectrum = gen_power_spectrum([params.freqs[0], params.freqs[-1]],
                                            [], params.get_params('peak'), 
                                            freq_res=params.freq_res, nlv=0)
    else:
        raise ValueError('Invalid component specified. Must be "both", "aperiodic", or "peak".')
    
    return spectrum


def params_to_df(params, max_peaks):
    """
    Convert FOOOFGroup object to pandas dataframe.

    Parameters
    ----------
    params : FOOOFGroup object
        FOOOFGroup object.
    max_peaks : int
        'max_n_peaks' parameter used to fit FOOOFGroup object.

    Returns
    -------
    df : pandas dataframe
        Pandas dataframe containing aperiodic parameters and gaussian parameters for each peak.
    """
    
    # imports
    import pandas as pd

    # get per params
    df_per = pd.DataFrame(params.get_params('peak'),
        columns=['cf','pw','bw','idx'])

    # get ap parmas
    if params.aperiodic_mode == 'knee':
        df_ap = pd.DataFrame(params.get_params('aperiodic'),  
            columns=['offset', 'knee', 'exponent'])
    elif params.aperiodic_mode == 'fixed':
        df_ap = pd.DataFrame(params.get_params('aperiodic'),  
            columns=['offset', 'exponent'])

    # get quality metrics
    df_ap['r_squared'] = params.get_params('r_squared')

    # initiate combined df
    df = df_ap.copy()
    columns = []
    for ii in range(max_peaks):
        columns.append([f'cf_{ii}',f'pw_{ii}',f'bw_{ii}'])
    df_init = pd.DataFrame(columns=np.ravel(columns))
    df = df.join(df_init)

    # app gaussian params for each peak fouond
    for i_row in range(len(df)):
        # check if row had peaks
        if df.index[ii] in df_per['idx']:
            # get peak info for row
            df_ii = df_per.loc[df_per['idx']==i_row].reset_index()
            # loop through peaks
            for i_peak in range(len(df_ii)):
                # add peak info to df
                for var_str in ['cf','pw','bw']:
                    df.at[i_row, f'{var_str}_{i_peak}'] = df_ii.at[i_peak, var_str]
    
    return df


def compute_adj_r2(params):
    """Calculate the adjusted r-squared for an existing FOOOF model.
    
    Parameters
    ----------
    params : FOOOF object
        FOOOF object that has been fit to data.

    Returns
    -------
    adj_r2 : float
        Adjusted r-squared value.
    """
    # imports
    from utils import adjust_r_squared
    
    # compute adjusted r-squared
    n_samples = len(params.freqs) # number of data points
    n_params = len(params.peak_params_) * 3 + 2 # number of parameters
    r_squared = params.get_params('r_squared')
    adj_r2 = adjust_r_squared(r_squared, n_params, n_samples)

    return adj_r2


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

