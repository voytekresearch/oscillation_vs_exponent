# -*- coding: utf-8 -*-
"""
Utility functions for working with specparam results/objects
"""

# Imports
import numpy as np


def compute_band_power(freq, spectra, band, method='mean', log_power=False):
    # imports
    from specparam.utils import trim_spectrum

    # get band of interest
    _, band = trim_spectrum(freq, spectra, band)

    # log-transform power
    if log_power:
        band = np.log10(band)

    # compute band power
    if method == 'mean':
        power = np.nanmean(band, axis=1)
    elif method == 'max':
        power = np.nanmax(band, axis=1)
    elif method == 'sum':
        power = np.nansum(band, axis=1)

    return power


def compute_adjusted_band_power(freq, spectra, params, band, **kwargs):
    # account for freq[0] = 0 Hz
    if freq[0] == 0:
        freq = freq[1:]
        spectra = spectra[:, 1:]

    # compute aperiodic component and subtract from spectra
    spectra_ap = params_to_spectra(params, component='aperiodic')
    freq_mask = np.logical_and(freq >= params.freqs[0], freq <= params.freqs[-1])
    spectra_adjusted = spectra[:, freq_mask] - spectra_ap

    # compute band power
    power = compute_band_power(freq[freq_mask], spectra_adjusted, band, 
                               **kwargs) 

    return power


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
    Extract aperiodic parameters from SpectralGroupModel object.

    Parameters
    ----------
    params : SpectralGroupModel object
        SpectralGroupModel object containing aperiodic parameters.

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
    from specparam import SpectralGroupModel

    # import specparam results
    params = SpectralGroupModel()
    params.load(fname)
    
    return extract_ap_params(params)


def params_to_spectra(params, component='both'):
    """
    Simulate aperiodic power spectra from SpectralGroupModel object.

    Parameters
    ----------
    params : SpectralGroupModel object
        SpectralGroupModel object containing aperiodic parameters.
    component : str
        Component to simulate ('both', 'aperiodic', or 'peak'). Default: 'both'.

    Returns
    -------
    spectra: array
        Aperiodic power spectra.

    """
    # imports
    from specparam.sim import sim_power_spectrum

    # init
    spectra = np.zeros([len(params), len(params.freqs)])
    
    # simulate aperiodic spectra for each
    if component == 'both':
        for ii in range(len(params)):
            _, spectra[ii] = sim_power_spectrum([params.freqs[0], 
                                                params.freqs[-1]],
                                                params[ii].aperiodic_params, 
                                                params[ii].peak_params, 
                                                freq_res=params.freq_res, nlv=0)
    elif component == 'aperiodic':
        for ii in range(len(params)):
            _, spectra[ii] = sim_power_spectrum([params.freqs[0], 
                                                 params.freqs[-1]],
                                                params[ii].aperiodic_params, [], 
                                                freq_res=params.freq_res, nlv=0)
    # simulate aperiodic spectra for each
    elif component == 'peak':
        for ii in range(len(params)):
            _, spectra[ii] = sim_power_spectrum([params.freqs[0], 
                                                 params.freqs[-1]],
                                                [], params[ii].peak_params, 
                                                freq_res=params.freq_res, nlv=0)
    else:
        raise ValueError('Invalid component specified. Must be "both", \
                         "aperiodic", or "peak".')
    
    return spectra


def params_to_spectrum(params, component='both'):
    """
    Simulate aperiodic power spectra from SpectralModel object.

    Parameters
    ----------
    params : SpectralModel object
        SpectralModel object containing aperiodic parameters.
    component : str
        Component to simulate ('both', 'aperiodic', or 'peak'). Default: 'both'.

    Returns
    -------
    spectra: array
        Aperiodic power spectra.

    """
    # imports
    from specparam.sim import sim_power_spectrum

    # simulate aperiodic spectra for each
    if component == 'both':
        _, spectrum = sim_power_spectrum([params.freqs[0], params.freqs[-1]],
                                            params.get_params('aperiodic'),
                                            params.get_params('peak'),
                                            freq_res=params.freq_res, nlv=0)
    elif component == 'aperiodic':
        _, spectrum = sim_power_spectrum([params.freqs[0], params.freqs[-1]],
                                            params.get_params('aperiodic'), [], 
                                            freq_res=params.freq_res, nlv=0)
    # simulate aperiodic spectra for each
    elif component == 'peak':
        _, spectrum = sim_power_spectrum([params.freqs[0], params.freqs[-1]],
                                            [], params.get_params('peak'), 
                                            freq_res=params.freq_res, nlv=0)
    else:
        raise ValueError('Invalid component specified. Must be "both", \
                         "aperiodic", or "peak".')
    
    return spectrum


def params_to_df(params, max_peaks):
    """
    Convert SpectralGroupModel object to pandas dataframe.

    Parameters
    ----------
    params : SpectralGroupModel object
        SpectralGroupModel object.
    max_peaks : int
        'max_n_peaks' parameter used to fit SpectralGroupModel object.

    Returns
    -------
    df : pandas dataframe
        Pandas dataframe containing aperiodic parameters and gaussian parameters 
        for each peak.
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
    """Calculate the adjusted r-squared for an existing SpectralModel.
    
    Parameters
    ----------
    params : SpectralModel object
        SpectralModel object that has been fit to data.

    Returns
    -------
    adj_r2 : float
        Adjusted r-squared value.
    """
    # imports
    from utils import adjust_r_squared
    
    # compute adjusted r-squared
    n_samples = len(params.freqs) # number of data points
    n_params = len(params.peak_params_) * 3 + len(params.aperiodic_params_) # number of parameters
    r_squared = params.get_params('r_squared')
    adj_r2 = adjust_r_squared(r_squared, n_params, n_samples)

    return adj_r2


def compute_intersection(params_0, params_1, return_spectra=False):
    """ 
    Calculate intersection of two spectra from SpectralModel objects.

    Parameters
    ----------
    params_0, params_1 : SpectralModel
        SpectralModel objects containing power spectra.
    return_spectra : bool, optional, default: False
        Whether to return the power spectra.    

    Returns
    -------
    intersection : 1d array
        intersection frequency
    intersection_idx : 1d array
        index of intersection frequency
    spectra_0, spectra_1 : 1d array
        Power spectra from each model, if requested.

    """

    # sim spectra from parameters
    spectra_0 = params_to_spectrum(params_0, component='aperiodic')
    spectra_1 = params_to_spectrum(params_1, component='aperiodic')

    # calc intersect of aperiodic spectra
    idx = np.argwhere(np.diff(np.sign(spectra_1 - spectra_0))).flatten()

    # account for no intersection or multiple intersections
    if not idx.any(): 
        intersection = np.nan
        intersection_idx = np.nan
    elif len(idx)==1: 
        intersection = params_0.freqs[np.squeeze(idx)]
        intersection_idx = np.squeeze(idx)
    elif len(idx)==2: 
        intersection = params_0.freqs[np.max(idx)]
        intersection_idx = np.max(idx)
    elif len(idx)==len(params_0.freqs):
        intersection = np.nan
        intersection_idx = np.nan

    if return_spectra:
        return intersection, intersection_idx, spectra_0, spectra_1
    else:
        return intersection, intersection_idx


def compute_intersections(params_0, params_1, return_spectra=False):
    """ 
    Calculate intersection of two spectra from SpectralGroupModel objects.

    Parameters
    ----------
    params_0, params_1 : SpectralGroupModel
        SpectralGroupModel objects containing power spectra.
    return_spectra : bool, optional, default: False
        Whether to return the power spectra.    

    Returns
    -------
    intersection : 1d array
        intersection frequency
    intersection_idx : 1d array
        index of intersection frequency
    spectra_0, spectra_1 : 1d array
        Power spectra from each model, if requested.

    """
    # imports
    from specparam import SpectralModel

    # Check inputs
    if not isinstance(params_0, SpectralModel) or not isinstance(params_1, SpectralModel):
        raise ValueError('Input must be SpectralModel of SpectralGroupModel objects.')
    n_chans = len(params_0)
    
    # Run analysis for SpectralModel input
    if n_chans == 1:
        results = compute_intersection(params_0, params_1, return_spectra)
        intersection, intersection_idx = results[:2]
        if return_spectra:
            spectra_0 = results[2]
            spectra_1 = results[3]

    # Run analysis for SpectralGroupModel input
    elif n_chans > 1:

        # check if input is same size
        if len(params_0) != len(params_1):
            raise ValueError('Input must be same size.')

        # initiate variables
        intersection = np.zeros([len(params_0)]) * np.nan
        intersection_idx = intersection.copy()
        spectra_0 = np.zeros([len(params_0), len(params_0.freqs)])
        spectra_1 = spectra_0.copy()

        # compute intersection for each channel
        for i_chan in range(len(params_0)):
            try:
                # get model parameters for channel
                params_0_i = params_0.get_model(i_chan)
                params_1_i = params_1.get_model(i_chan)

                # compute intersection
                results = compute_intersection(params_0_i, params_1_i, return_spectra)
                intersection[i_chan], intersection_idx[i_chan] = results[:2]
                if return_spectra:
                    spectra_0[i_chan] = results[2]
                    spectra_1[i_chan] = results[3]

            # return nan if no model parameters are available
            except:
                intersection[i_chan], intersection_idx[i_chan] = np.nan, np.nan
                if return_spectra:
                    spectra_0[i_chan] = [np.nan] * len(params_0.freqs)
                    spectra_1[i_chan] = [np.nan] * len(params_0.freqs)
                continue
    else:
        raise ValueError('Check size of input.')

    # return results
    if return_spectra:
        return intersection, intersection_idx, spectra_0, spectra_1
    else:
        return intersection, intersection_idx


def save_report_sm(sm, file_name, file_path=None, plot_peaks=None, plot_aperiodic=True, plt_log=True, 
                    add_legend=True, data_kwargs=None, model_kwargs=None, aperiodic_kwargs=None, 
                    peak_kwargs=None, show_fig=False):
    """ Modified from FOOOF.core.reports.save_report_fm
    Generate and save out a PDF report for a power spectrum model fit.

    Parameters
    ----------
    sm : SpectralModel
        Object containing a power spectrum and (optionally) results from fitting.    
    file_name : str
        Name to give the saved out file.    
    file_path : str, optional
        Path to directory to save to. If None, saves to current directory.    
    plot_peaks : None or {'shade', 'dot', 'outline', 'line'}, optional
        What kind of approach to take to plot peaks. If None, peaks are not specifically plotted.
        Can also be a combination of approaches, separated by '-', for example: 'shade-line'.
    plot_aperiodic : boolean, optional, default: True
        Whether to plot the aperiodic component of the model fit.
    plt_log : boolean, optional, default: True
        Whether to plot the frequency values in log10 spacing.
    add_legend : boolean, optional, default: False
        Whether to add a legend describing the plot components.
    data_kwargs, model_kwargs, aperiodic_kwargs, peak_kwargs : None or dict, optional
        Keyword arguments to pass into the plot call for each plot element.
    show_fig : bool, optional, default: False
        Whether to show the plot. If False, the plot is closed after saving.
    """
    # imports 
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from specparam.core.strings import gen_settings_str, gen_model_results_str
    from specparam.core.io import fname, fpath

    # settings
    REPORT_FIGSIZE = (16, 20)
    REPORT_FONT = {'family': 'monospace',
                'weight': 'normal',
                'size': 16}
    SAVE_FORMAT = 'pdf'

    # Set up outline figure, using gridspec
    _ = plt.figure(figsize=REPORT_FIGSIZE)
    grid = gridspec.GridSpec(3, 1, height_ratios=[0.45, 1.0, 0.25])

    # First - text results
    ax0 = plt.subplot(grid[0])
    results_str = gen_model_results_str(sm)
    ax0.text(0.5, 0.7, results_str, REPORT_FONT, ha='center', va='center')
    ax0.set_frame_on(False)
    ax0.set_xticks([])
    ax0.set_yticks([])

    # Second - data plot
    ax1 = plt.subplot(grid[1])
    sm.plot(plot_peaks=plot_peaks, plot_aperiodic=plot_aperiodic, plt_log=plt_log, add_legend=add_legend,
            ax=ax1, data_kwargs=data_kwargs, model_kwargs=model_kwargs, aperiodic_kwargs=aperiodic_kwargs, 
            peak_kwargs=peak_kwargs)

    # Third - specparam settings
    ax2 = plt.subplot(grid[2])
    settings_str = gen_settings_str(sm, False)
    ax2.text(0.5, 0.1, settings_str, REPORT_FONT, ha='center', va='center')
    ax2.set_frame_on(False)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Save out the report (and optionally show)
    plt.savefig(fpath(file_path, fname(file_name, SAVE_FORMAT)))
    if show_fig:
        plt.show()
    else:
        plt.close()


def print_report_from_group(params, i_model, fname_out, show_fig=False):
    """
    Generate and save out a PDF report for a power spectrum model fit within a
    SpectralGroupModel object.

    Parameters
    ----------
    params : SpectralGroupModel
        Object with results from fitting a group of power spectra.
    i_model : int
        Index of the model for which to generate a report.
    fname_out : str
        Name to give the saved out file.
    show_fig : bool, optional, default: False
        Whether to show the plot. If False, the plot is closed after saving.
    
    """
    # imports
    from specparam import SpectralGroupModel
    from specparam.sim.gen import gen_aperiodic, gen_periodic

    # create specparam object and add settings
    sm = SpectralGroupModel()
    sm.add_settings(params.get_settings())

    # Copy results for model of interest and additional data needed for plotting
    sm.add_results(params[i_model])
    sm.power_spectrum = params.power_spectra[i_model]
    sm.freq_range = params.freq_range
    sm.freq_res = params.freq_res
    sm.freqs = params.freqs

    # generate and perioidc/aperiodic fits from parameters
    sm._ap_fit = gen_aperiodic(params.freqs, params[i_model].aperiodic_params)
    sm._peak_fit = gen_periodic(params.freqs, np.ndarray.flatten(params[i_model].gaussian_params))
    sm.modeled_spectrum_ = sm._ap_fit + sm._peak_fit

    # save report
    save_report_sm(sm, fname_out, plt_log=True, show_fig=show_fig)
