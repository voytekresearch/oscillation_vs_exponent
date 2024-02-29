# -*- coding: utf-8 -*-
"""
Stistical utility functions
"""

# Imports
import numpy as np


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

    # shuffle conditions
    order = gen_random_order(n_iter, len(data_a)*2)
    surrogate_0, surrogate_1 = shuffle_arrays(data_a, data_b, order)

    # average values for each surrogate condition and compute difference
    means_0 = np.nanmean(surrogate_0, axis=1)
    means_1 = np.nanmean(surrogate_1, axis=1)
    distr = means_1 - means_0 # surrogate distribution

    # compute p-value
    diff = np.nanmean(data_b) - np.nanmean(data_a) # true difference
    p_val = comp_resampling_pval(distr, diff)

    return p_val


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
    """

    # calc 2-sided p value
    n_iterations = np.size(distribution)
    n_more = np.sum(np.abs(distribution) > np.abs(value))
    p_value = n_more / n_iterations
        
    return p_value


def shuffle_rows(matrix_a, matrix_b, order):
    """
    Shuffling rows between two matrices.

    Parameters
    ----------
    matrix_a, matrix_b : numpy.ndarray
        2D matrices to shuffle.
    order : numpy.ndarray
        Array of shape (n_iterations, length) containing random orders of
        integers from 0 to length, where n_interations is the number of
        shuffles and length is the number of rows in eahc matrix.

    Returns
    -------
    matrix_as, matrix_bs : numpy.ndarray
        Shuffled matrices.
    """
    
    # concatenate matrices
    matrix_ab = np.vstack([matrix_a, matrix_b])

    # shuffles rows, then split matrix into 2 groups
    n_iter = order.shape[0]
    n_rows = matrix_a.shape[0]
    matrix_as = np.zeros([n_iter,*matrix_a.shape])
    matrix_bs = np.zeros([n_iter,*matrix_a.shape])
    for i_iter in range(n_iter):
        matrix_shuffled = matrix_ab[order[i_iter]]
        matrix_as[i_iter] = matrix_shuffled[:n_rows]
        matrix_bs[i_iter] = matrix_shuffled[n_rows:]
    
    return matrix_as, matrix_bs


def resample_means(data1, data2, surrogate_runs):
    """
    Resample means between two groups of data.
    """

    # data sizes
    data1_size = np.size(data1)
    data2_size = np.size(data2)

    # real differnce in means between datasets
    real_difference = np.mean(data1) - np.mean(data2)

    # pooled data for resampling analyses
    pooled_data = np.append(data1, data2)

    surr_difference = np.zeros(surrogate_runs)
    for i in range(surrogate_runs):
        # randomly permute the pooled data
        permutation_array = np.random.permutation(data1_size + data2_size)
        permuted_data = pooled_data[permutation_array]

        # sample from the pooled data
        surr_data1 = permuted_data[:data1_size]
        surr_data2 = permuted_data[data1_size:]

        # build distrubution of differenes of permuted data means
        surr_difference[i] = np.mean(surr_data1) - np.mean(surr_data2)
    
    # where, along the distribution of the surrogate data, does the real data lie?
    exact_p_value = np.count_nonzero(np.abs(real_difference) < np.abs(surr_difference)) / surrogate_runs

    return real_difference, surr_difference, exact_p_value
