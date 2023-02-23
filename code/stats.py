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
    matrix_ab = np.vstack([matrix_a, matrix_b])

    # split matrix into 2 groups
    n_iter = order.shape[0]
    n_rows = matrix_a.shape[0]
    matrix_as = np.zeros([n_iter,*matrix_a.shape])
    matrix_bs = np.zeros([n_iter,*matrix_a.shape])
    for i_iter in range(n_iter):
        matrix_shuffled = matrix_ab[order[i_iter]]
        matrix_as[i_iter] = matrix_shuffled[:n_rows]
        matrix_bs[i_iter] = matrix_shuffled[n_rows:]
    
    return matrix_as, matrix_bs

def shuffle_arrays(array_a, array_b, order):
    """
    Perform resampling by shuffling samples of 2 arrays.

    Parameters
    ----------
    array_a, array_b : numpy.ndarray
        Arrays to shuffle.
    order : numpy.ndarray
        Array of shape (n_iterations, length) containing random orders of
        integers from 0 to length, where n_interations is the number of
        shuffles and length is the number of rows in eahc matrix.

    Returns
    -------
    matriarray_as, array_bs : numpy.ndarray
        Shuffled arrays.
    """
    
    # concatenate 2 groups 
    array_ab = np.concatenate([array_a, array_b])

    # initialize arrays
    n_iter = int(len(order))
    array_as = np.zeros([n_iter, len(array_a)])
    array_bs = np.zeros([n_iter, len(array_a)])

    # shuffle samples
    n_samples = int(len(array_a))
    for i_iter in range(n_iter):
        array_shuffled = array_ab[order[i_iter]]
        array_as[i_iter] = array_shuffled[:n_samples]
        array_bs[i_iter] = array_shuffled[n_samples:]
    
    return array_as, array_bs

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

def run_resampling_analysis(data_a, data_b , n_iter):
    """
    Run resampling analysis.

    Parameters
    ----------
    data_a, data_b : numpy.ndarray
        Data to compare.
    n_iter : int
        Number of iterations/shuffles.

    Returns
    -------
    p_val : float
        P-value for resampling analysis.
    sign : int
        Sign of effect (1   :   positive (a<b), 
                        -1  :   negative (a>b), 
                        0   :   no difference (a=b)).
    """

    # get random order for trials
    order = gen_random_order(n_iter, len(data_a)*2)

    # shuffle conditions
    shuffled_a, shuffled_b = shuffle_arrays(data_a, data_b, order)

    # average shuffled power values over time windows and trials, then compute difference
    mean_a = np.nanmean(shuffled_a, axis=1)
    mean_b = np.nanmean(shuffled_b, axis=1)
    distr = mean_a - mean_b

    # compute p value
    diff = np.nanmean(data_a) - np.nanmean(data_b)
    p_val, sign = comp_resampling_pval(distr, diff)

    return p_val, sign

