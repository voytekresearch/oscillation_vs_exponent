"""
This module contains functions for performing the hierarchical bootstrap. 

The hierarchical bootstrap is a non-parametric method for testing whether the means
of two distributions are significantly different. This method is an extension of the standard
bootstrap method that accounts for non-independence in the data resulting from a hierarchical
structure, e.g. if data are recorded from multiple electrodes within each subject, 
and multiple subjects within each experimental condition.
"""


# imports
import numpy as np

def run_hierarchical_bootstrap(df, variable, condition, level_1, level_2, n_iterations=1000,
                               verbose=True, plot=True, **kwargs):    
    """
    Perform hierarchical bootstrap. This function performs a hierarchical bootstrap
    to test whether the means of two distributions are significantly different. 

    NOTE: a p-value of 0.5 indicates that the two distributions are identical; a
    p-value close to 0 indicates that distributions_0 is greater than distributions_1;
    and a p-value close to 1 indicates that distributions_1 is greater than distributions_0.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing data to resample.
    variable : str
        Variable to resample.
    condition : str
        Experimental condition of interest.
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
    **kwargs : dict
        Additional keyword arguments for plotting.

    Returns
    -------
    p_value : float
        p-value for difference between conditions.
    distribution_0 : numpy.ndarray
        Resampled distribution for condition 0.
    distribution_1 : numpy.ndarray
        Resampled distribution for condition 1.
    """

    # split groups
    df_0, df_1 = _split_experimental_conditions(df, condition)

    # run bootstrap
    distribution_0 = _hierarchical_bootstrap(df_0, variable, level_1, level_2, n_iterations)
    distribution_1 = _hierarchical_bootstrap(df_1, variable, level_1, level_2, n_iterations)

    # compute p-boot 
    p_value, joint_prob, bin_edges = _compute_p_value(distribution_0, distribution_1)

    # print/plot results    
    if verbose:
        print(f"p-value: {p_value:.3f}")
    if plot:
        _plot_bootstrap_results(df, variable, condition, distribution_0, distribution_1,
                               joint_prob, bin_edges, **kwargs)

    # return p_value, distribution_0, distribution_1
    return p_value, joint_prob, bin_edges, distribution_0, distribution_1


def _split_experimental_conditions(df, condition):
    """
    Split dataframe into two groups based on experimental condition.
    """

    # check that there are only two experimental conditions
    conditions = np.sort(df[condition].unique())
    if len(conditions) != 2:
        raise ValueError("More than two experimental conditions detected.")
        

    # split dataframe by experimental condition
    df_0 = df.loc[df[condition]==conditions[0]]
    df_1 = df.loc[df[condition]==conditions[1]]

    return df_0, df_1


def _hierarchical_bootstrap(df, variable, level_1, level_2, iterations):
    """
    Get distribution of resampled means for hierarchical bootstrap. This function
    resamples the data and computes the mean for each iteration of the
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
        instances_per_cluster[i_cluster] = len(df.loc[df[level_1]==cluster_i, level_2].unique())
    n_instances = int(np.nanmean(instances_per_cluster)) # use average number of instances per cluster

    # loop through iterations
    distribution = np.zeros(iterations)
    for i_iteration in range(iterations):
        # Resample level 2 
        clusters_resampled = np.random.choice(clusters, size=n_clusters)

        # resample level 3 and get data for each cluster
        values = []
        for i_cluster, cluster_i in enumerate(clusters_resampled):
            # resample level 3
            instances = df.loc[df[level_1]==cluster_i, level_2].unique()
            instances_resampled = np.random.choice(instances, size=n_instances)

            # get data for each instance within cluster and average
            for i_instance, instance_i in enumerate(instances_resampled):
                value = df.loc[(df[level_1]==cluster_i) & (df[level_2]==instance_i), variable].values[0]
                values.append(value)

        # compute average for iteration
        distribution[i_iteration] = np.nanmean(values)

    return distribution


def _compute_p_value(distribution_0, distribution_1, n_bins=30):    
    '''
    Compute the p-value for the hierarchical bootstrap. This function computes
    the joint probability of the two distributions and then sums the upper
    triangle of the joint probability matrix to get the p-value. A p-value of
    0.5 indicates that the two distributions are identical; a p-value close to 0
    indicates that distributions_0 is greater than distributions_1; and a p-value
    close to 1 indicates that distributions_1 is greater than distributions_0.

    This function is based on Saravanan et al. 2020 (https://github.com/soberlab/Hierarchical-Bootstrap-Paper)
    '''

    # calculate probabilities for each distribution
    all_values = np.concatenate([distribution_0, distribution_1])
    bin_edges = np.linspace(np.min(all_values), np.max(all_values), n_bins)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_edges = np.append(bin_edges, bin_edges[-1] + bin_width) - (bin_width/2) # add last bin edge and shift by half bin width
    prob_0 = np.histogram(distribution_0, bins=bin_edges)[0] / len(distribution_0)
    prob_1 = np.histogram(distribution_1, bins=bin_edges)[0] / len(distribution_1)

    # compute joint probability
    joint_prob = np.outer(prob_0, prob_1)
    joint_prob = joint_prob / np.sum(joint_prob) # normalize

    # compute p-value
    p_value = np.sum(np.triu(joint_prob))

    return p_value, joint_prob, bin_edges


def _plot_bootstrap_results(df, variable, condition, distribution_0, distribution_1,
                           joint_prob, bin_edges, colors=['k','b'], 
                           labels=['0', '1']):
    """
    Plot bootstrap results. PLotting function for run_hierarchical_bootstrap().
    """

    # imports
    import matplotlib.pyplot as plt

    # create figure
    fig, (ax0, ax1, ax2) = plt.subplots(1,3, figsize=(18,4))

    # ax0: plot orignal distributions
    conditions = np.sort(df[condition].unique())
    data_0 = df.loc[df[condition]==conditions[0], variable].values
    data_1 = df.loc[df[condition]==conditions[1], variable].values
    print(data_0.shape, data_1.shape)
    bin_edges_ = np.linspace(np.nanmin([data_0, data_1]), np.nanmax([data_0, data_1]), 30)
    ax0.hist(data_0.ravel(), bins=bin_edges_, color=colors[0], alpha=0.5, label=labels[0])
    ax0.hist(data_1.ravel(), bins=bin_edges_, color=colors[1], alpha=0.5, label=labels[1])
    ax0.set_xlabel('value')
    ax0.set_ylabel('count')
    ax0.set_title('Original dataset')
    
    # ax1: plot distributions
    ax1.hist(distribution_0, bins=bin_edges, color='k', alpha=0.5, label=conditions[0])
    ax1.hist(distribution_1, bins=bin_edges, color='b', alpha=0.5, label=conditions[1])
    ax1.set_xlabel(variable)
    ax1.set_ylabel('count')
    ax1.set_title('Bootstrap results')
    ax1.legend()

    # ax2: plot joint probability
    im = ax2.pcolormesh(bin_edges, bin_edges, joint_prob, cmap='hot')
    ax2.set_xlabel(conditions[0])
    ax2.set_ylabel(conditions[1])
    ax2.set_title('Joint probability')
    fig.colorbar(im, ax=ax2)

