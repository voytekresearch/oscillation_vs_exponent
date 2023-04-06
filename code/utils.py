# -*- coding: utf-8 -*-
"""
Utility functions
"""

# Imports
import numpy as np


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

    hours = int(np.floor(duration / 3600))
    mins = int(np.floor(duration%3600 / 60))
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