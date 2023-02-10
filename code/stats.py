# -*- coding: utf-8 -*-
"""
Stistical utility functions
"""

# Imports
import numpy as np

def gen_random_order(n_iterations, length):
    """
    Generate random order of integers (for resampling).
    
    Parameters
    ----------
    n_iterations : int
        Number of iterations (shuffles/orders) to return.
    length : int
        Length of each shuffle/order.

    Returns
    -------
    order : numpy.ndarray
        Array of shape (n_iterations, length) containing random orders of
        integers from 0 to length.
    """
    # generate pseudo-random order for trials
    order = np.zeros([n_iterations, length])
    for i_iter in range(n_iterations):
        order[i_iter] = np.random.permutation(np.linspace(0, length-1, length))
    order = order.astype(int)
    
    return order

def shuffle_rows(matrix_a, matrix_b, order):
    """
    Perform resampling by shuffling rows of 2 matrices.

    Parameters
    ----------
    matrix_a, matrix_b : numpy.ndarray
        Matrices to shuffle.
    order : numpy.ndarray
        Array of shape (n_iterations, length) containing random orders of
        integers from 0 to length, where n_interations is the number of
        shuffles and length is the number of rows in eahc matrix.

    Returns
    -------
    matrix_as, matrix_bs : numpy.ndarray
        Shuffled matrices.
    """
    
    # concatenate 2 groups of spectra and shuffle rows
    matrix_ab = np.concatenate([matrix_a, matrix_b])[order]

    # split matrix into 2 groups
    n_rows = matrix_a.shape[0]
    matrix_as = matrix_ab[:n_rows]
    matrix_bs = matrix_ab[n_rows:]
    
    return matrix_as, matrix_bs

def comp_resampling_pval(distribution, value):
    """
    Compute p-value for resampling analysis.

    Parameters
    ----------
    distribution : numpy.ndarray
        Distribution of values obtained through resampling.
    value : float
        Value to compare to distribution.

    Returns
    -------
    p_value : float
        P-value for resampling analysis.
    sign : int
        Sign of effect (1 = positive, -1 = negative, 0 = no effect).
    """

    n_iterations = np.size(distribution)
    n_less = np.sum(distribution < value)
    n_more = np.sum(distribution > value)
    
    # calc 2-sided p value
    p_value = np.min([n_less, n_more]) / n_iterations * 2
    
    # determine direction of effect
    if n_less < n_more: 
        sign = -1
    elif n_less > n_more: 
        sign = 1
    elif n_less == n_more: 
        sign = 0
        
    return p_value, sign