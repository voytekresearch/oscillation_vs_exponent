"""
This module contains functions for performing the hierarchical bootstrap. 

The hierarchical bootstrap is a non-parametric method for computing the 
uncertainty of statistics derived from nested data structure. For instance, this 
method can be used to test whether the means of two distributions are 
significantly different from one another (as implemented here). This method is 
an extension of the standard bootstrap method that accounts for 
non-independence in the dataset resulting from a nested, multi-level, or 
hierarchical structure, e.g. if data are recorded from multiple electrodes 
within each subject, within each experimental condition.

A great overview of the hierarchical bootstrap is provided in "Application of 
the hierarchical bootstrap to multi-level data in neuroscience" by Saravanan et 
al., 2020; available: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7906290/. The 
authors were generous enough to provide code for performing the hierarchical 
bootstrap here: https://github.com/soberlab/Hierarchical-Bootstrap-Paper. The 
present module extends the aforementioned implementation by providing support
for datasets in which data are not equally distributed across conditions and/or 
clusters. Furthermore, this module provides additional functionality for
plotting of the results.
"""


# imports
import numpy as np
import matplotlib.pyplot as plt

def onesample_hierarchical_bootstrap(df, variable, level_1, level_2, n_iterations=1000, 
                           verbose=True, plot=True):    
    """
    Perform the one-sample hierarchical bootstrap. 

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing data to resample.
    variable : str
        Variable to resample.
    level_1 : str
        First level of hierarchy to resample.
    level_2 : str
        Second level of hierarchy to resample.
    iterations : int
        Number of iterations for resampling.
    verbose : bool
        Whether to print p-value.
    plot : bool
        Whether to plot results.

    Returns
    -------
    p_value : float
        p-value for one-sample test.
    distribution : numpy.ndarray
        Resampled distribution.
    """

    # perform hierarchical bootstrap
    distribution = _hierarchical_bootstrap(df, variable, level_1, level_2, 
                                           n_iterations)

    # compute p-value
    p_value = _compute_p_value(distribution)

    # print/plot results
    if verbose:
        print(f"p-value: {p_value:.3f}")
    if plot:
        _plot_results(df, variable, distribution)

    return p_value, distribution


def _hierarchical_bootstrap(df, variable, level_1, level_2, iterations):
    """
    Get distribution of resampled means for hierarchical bootstrap. This 
    function resamples the data and computes the mean for each iteration of the
    bootstrap.

    NOTE: the number of instances per cluster is computed as the average number
    of instances per cluster. This is done to avoid biasing the resampling
    towards clusters with more instances.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing data to resample.
    variable : str
        Variable to resample.
    level_1 : str
        First level of hierarchy to resample.
    level_2 : str
        Second level of hierarchy to resample.
    iterations : int
        Number of iterations for resampling.

    Returns
    -------
    distribution : numpy.ndarray
        Resampled distribution.

    """

    # get cluster info
    clusters = df[level_1].unique()
    n_clusters = len(clusters)

    # count number of instances per cluster
    instances_per_cluster = np.zeros(n_clusters)
    for i_cluster, cluster_i in enumerate(clusters):
        instances_per_cluster[i_cluster] = len(df.loc[df[level_1]==cluster_i, 
                                                      level_2].unique())
    n_instances = int(np.nanmean(instances_per_cluster)) # use average number of instances per cluster

    # Precompute unique instances for each cluster
    cluster_instance_map = {cluster: df.loc[df[level_1] == cluster, level_2].unique() for cluster in clusters}

    # loop through iterations
    distribution = np.zeros(iterations)
    for i_iteration in range(iterations):
        # resample level 1 (clusters)
        clusters_resampled = np.random.choice(clusters, size=n_clusters)

        # resample level 2 (instances) and get data for each cluster
        values = []
        for i_cluster, cluster_i in enumerate(clusters_resampled):
            # resample level 3
            instances = cluster_instance_map[cluster_i]
            instances_resampled = np.random.choice(instances, size=n_instances)

            # get data for each instance within cluster
            for _, instance_i in enumerate(instances_resampled):
                value = df.loc[(df[level_1]==cluster_i) & \
                               (df[level_2]==instance_i), variable].values[0]
                values.append(value)

        # compute average for iteration
        distribution[i_iteration] = np.nanmean(values)

    return distribution


def _compute_p_value(distribution):
    """
    Compare distribution to zero.
    """

    # compute p-value
    n_more = np.sum(distribution > 0)
    n_less = np.sum(distribution < 0)
    p_value = np.min([n_more, n_less]) / len(distribution)
    
    return p_value


def _plot_results(df, variable, distribution):
    """
    Plot bootstrap results. PLotting function for run_hierarchical_bootstrap().
    """

    # create figure
    _, (ax0, ax1) = plt.subplots(1,2, figsize=(12, 4))

    # ax0: plot orignal distributions
    data = df[variable].values
    bin_edges = np.linspace(np.nanmin(data), np.nanmax(data), 30)
    ax0.hist(data.ravel(), bins=bin_edges, color='k')
    ax0.set_xlabel('value')
    ax0.set_ylabel('count')
    ax0.set_title('Original dataset')
    
    # ax1: plot reasmapled distributions
    ax1.hist(distribution, bins=bin_edges, color='k')
    ax1.set_xlabel(variable)
    ax1.set_ylabel('count')
    ax1.set_title('Bootstrap results')

