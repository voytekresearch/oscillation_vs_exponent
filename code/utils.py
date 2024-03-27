# -*- coding: utf-8 -*-
"""
Utility functions
"""

# Imports
import numpy as np


def confidence_interval(data, confidence=0.95, axis=0):
    """ 
    Compute confidence interval for data.

    Parameters
    ----------
    data : array
        Data to compute confidence interval for.
    confidence : float
        Confidence level for confidence interval.
    axis : int
        Axis to compute confidence interval over.

    Returns
    -------
    ci : array
        Confidence interval.
    """

    from scipy.stats import norm

    mean = np.nanmean(data, axis=axis)
    std = np.nanstd(data, axis=axis)
    ci = norm.interval(confidence, loc=mean, scale=std/np.sqrt(data.shape[axis]))

    return ci


def get_start_time():
    from time import time

    return time()


def print_time_elapsed(t_start):
    from time import time

    hour, min, sec = hour_min_sec(time() - t_start)
    print(f"Time elapsed: {hour} hour, {min} min, and {sec :0.1f} s")


def hour_min_sec(duration):
    """
    Convert duration in seconds to hours, minutes, and seconds.

    Parameters
    ----------
    duration : float
        Duration in seconds.

    Returns
    -------
    hours, mins, secs : int
        Duration in hours, minutes, and seconds.
    """

    hours = int(duration // 3600)
    mins = int(duration % 3600 // 60)
    secs = int(duration % 60)
    
    return hours, mins, secs


def adjust_r_squared(r_squared, n_params, n_samples):
    """
    Compute adjusted r-squared.
    
    Parameters
    ----------
    r_squared : float
        R-squared value.
    n_params : int
        Number of parameters in model.
    n_data : int
        Number of data points in model.
        
    Returns
    -------
    adjusted_r_squared : float
        Adjusted r-squared value.
    """
    
    adjusted_r_squared = 1 - ((1 - r_squared) * (n_samples - 1)) / (n_samples - n_params - 1)

    return adjusted_r_squared


def diff_spectra(spectra_a, spectra_b):
    """
    Compute the difference between two arrays of power spectra.
    Spectra are log transformed before computing the difference.
    
    Parameters
    ----------
    spectra_a : array
        Array of power spectra.
    spectra_b : array
        Array of power spectra.

    Returns
    -------
    spectra : array
        Difference between spectra_a and spectra_b.
    
    """
    
    # log transform and compute difference
    spectra = np.log(spectra_b) - np.log(spectra_a)

    return spectra


def combine_images(dir_fig, fname_out):
    """
    Combine images within a folder into a single figure and save. This 
    function is used to combine brain images of each hemisphere (and view i.e.
    lateral and medial) into a single figure.

    """
    # imports
    import os
    import matplotlib.pyplot as plt

    # load images and join
    images = [plt.imread(f"{dir_fig}/{f}") for f in os.listdir(f"{dir_fig}")]
    image = np.concatenate(images, axis=1)

    # save figure
    fig, ax = plt.subplots(1,1, figsize=(20, 10))
    ax.imshow(image)
    ax.axis('off')
    fig.savefig(fname_out, bbox_inches='tight', dpi=300)  

