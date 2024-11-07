"""
This module contains functions for performing the paired hierarchical bootstrap. 

The hierarchical bootstrap is a non-parametric method to estimate the 
uncertainty of statistics derived from nested data structures. This method is an 
extension of the standard bootstrap method that accounts for non-independence 
in the dataset resulting from a nested, multi-level, or hierarchical structure, 
e.g. if data are recorded from multiple electrodes within each subject, within 
each experimental condition. Furthermore, the paired hierarchical bootstrap 
additionally accounts for paired data structures, e.g. if data are recorded 
before and after an intervention for each subject.
"""


# imports
import numpy as np
from stats import mean_difference
import matplotlib.pyplot as plt
import seaborn as sns

def hierarchical_bootstrap(df, variable, condition, level_1, level_2, 
                               n_iterations=1000, verbose=True, plot=True):    
    """
    Perform paired hierarchical bootstrap. This function performs a paired 
    hierarchical bootstrap to test whether the means of two distributions are
    significantly different.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing data to resample.
    variable : str
        Variable to resample. This is the dependent variable.
    condition : str
        Experimental condition of interest. This is the independent variable.
    level_1 : str
        First level of hierarchy to resample. This is the higher level i.e.
        level_2 is nested within level_1; e.g. electrodes within subjects.
    level_2 : str
        Second level of hierarchy to resample. This is the lower level i.e.
        level_2 is nested within level_1; e.g. electrodes within subjects.
    iterations : int
        Number of iterations for resampling.
    verbose : bool
        Whether to print results, including p-value and true differnce.
    plot : bool
        Whether to plot results, including empirical data and resampled
        distribution.
    **kwargs : dict
        Additional keyword arguments for plotting.

    Returns
    -------
    p_value : float
        p-value for difference between conditions.
    sign : int
        Sign of effect (1   :   positive (condition_1 > condition_0)
                        -1  :   negative (condition_0 > condition_1)
    distribution : numpy.ndarray
        Distribution of resampled test statistic.
    true_mean : float
        True mean difference between conditions.
    """

    # run bootstrap
    distribution = _hierarchical_bootstrap(df, variable, condition, level_1, 
                                           level_2, n_iterations)

    # compute p-boot 
    p_value, sign = _compute_p_value(distribution)

    # compute true mean difference
    conditions = df[condition].unique()
    df_pivot = df.pivot(index=[level_1, level_2], columns=condition, 
                        values=variable).reset_index()
    df_pivot['difference'] = df_pivot[conditions[1]] - df_pivot[conditions[0]]
    true_mean = np.nanmean(df_pivot['difference'])

    # print results
    if verbose:
        _print_results(p_value, true_mean, n_iterations, conditions)

    # plot results
    if plot:
        _plot_results(df, variable, condition, level_1, level_2, distribution)

    return p_value, sign, distribution, true_mean


def _hierarchical_bootstrap(df, variable, condition, level_1, level_2, 
                            iterations):
    """
    Perform paired hierarchical bootstrap. This function resamples the data,
    taking into account the paired and hierarchical structure of the dataset. 
    The mean difference between the two conditions is then computed as the 
    test statistic.

    NOTE: the number of instances (level_2) per cluster (level 1) is computed 
    as the average number of instances per cluster. This is done to avoid 
    biasing the resampling towards clusters with more instances.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing data to resample.
    variable : str
        Variable to resample.
    condition : str
        Experimental condition of interest.
    level_1 : str
        First level of hierarchy to resample (higher level).
    level_2 : str
        Second level of hierarchy to resample (lower level).
    iterations : int
        Number of iterations for resampling.

    Returns
    -------
    distribution : numpy.ndarray
        Distribution of the resampled test statistic.

    """

    # check input data
    df = check_input(df, variable, condition, level_1, level_2)

    # get cluster and condition info
    clusters = df[level_1].unique()
    n_clusters = len(clusters)
    conditions = df[condition].unique()
    if len(conditions) != 2:
        raise ValueError("Condition must have two unique values.")

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
        values = np.zeros([n_clusters, n_instances, 2])
        for i_cluster, cluster_i in enumerate(clusters_resampled):
            # resample level 3
            instances = df.loc[df[level_1]==cluster_i, level_2].unique()
            instances_resampled = np.random.choice(instances, size=n_instances)

            # get data for each instance within cluster and average
            for i_instance, instance_i in enumerate(instances_resampled):
                values_ii = df.loc[(df[level_1]==cluster_i) & 
                                  (df[level_2]==instance_i)]
                for i_condtion, condition_i in enumerate(conditions):
                    value = values_ii.loc[values_ii[condition]==condition_i, variable].values
                    values[i_cluster, i_instance, i_condtion] = value
                
        # compute average for iteration
        distribution[i_iteration] = mean_difference(values[...,0], values[...,1])

    return distribution


def check_input(df, variable, condition, level_1, level_2):
    """
    Check input data for paired hierarchical bootstrap. This function checks 
    that each instance has both conditions present, and that the data is
    structured as a nested, hierarchical dataset.

    """
    
    # check that 'variable', 'condition', 'level_1', and 'level_2' are in df
    if variable not in df.columns:
        raise ValueError(f"Variable '{variable}' not found in dataframe.")
    if condition not in df.columns:
        raise ValueError(f"Condition '{condition}' not found in dataframe.")
    if level_1 not in df.columns:
        raise ValueError(f"Level 1 '{level_1}' not found in dataframe.")
    if level_2 not in df.columns:
        raise ValueError(f"Level 2 '{level_2}' not found in dataframe.")

    # check that each level_1-level_2 pair has data for both conditions
    # and drop cases of missing data
    n_dropped = 0
    n_instances = 0
    clusters = df[level_1].unique()
    for cluster in clusters:
        instances = df.loc[df[level_1]==cluster, level_2].unique()
        for instance in instances:
            df_i = df.loc[(df[level_1]==cluster) & (df[level_2]==instance)]
            if (len(df_i) != 2) or (len(df_i[condition].unique()) != 2) or \
                (df_i[variable].isnull().any()):
                df = df.drop(df_i.index)
                n_dropped += 1
            n_instances += 1

    if n_dropped > 0:
        print(f"Warning: {n_dropped}/{n_dropped+n_instances} instances dropped due to missing data.")

    return df


def _compute_p_value(distribution):    
    '''
    Compute the p-value for the paired hierarchical bootstrap. This function 
    computes the p-value for the null hypothesis that the means of two
    conditions are equal.

    '''

    # count values greater than 0
    n_greater = np.sum(distribution > 0)
    n_less = np.sum(distribution < 0)
    p_value = np.min([n_greater, n_less]) / len(distribution)
    sign = np.sign(n_greater - n_less)

    return p_value, sign


def _plot_results(df, variable, condition, level_1, level_2, distribution):
    """
    Plot paired hierarchical bootstrap results. This function plots the original 
    data and the resampled distribution of the test statistic. The original data
    is plotted as a violin plot with swarm plot overlayed, and paired data 
    points connected. The resampled distribution is plotted as a histogram with
    the true mean difference and null hypothesis (0) annotated.
    """

    # pivot data
    df_pivot = df.pivot(index=[level_1, level_2], columns=condition, 
                        values=variable).reset_index()

    # set plotting params
    plotting_params = {
        'data'  :   df,
        'x'     :   condition,
        'y'     :   variable,
    }

    # create figure
    _, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,4))

    # ax0: plot empirical data
    sns.violinplot(**plotting_params, ax=ax0, palette='colorblind')
    sns.swarmplot(**plotting_params, ax=ax0, size=2, palette='dark:#000000')
    conditions = df[condition].unique()
    for i_chan in range(df_pivot.shape[0]):
        y = [df_pivot.loc[i_chan, conditions[0]], 
            df_pivot.loc[i_chan, conditions[1]]]
        ax0.plot([0, 1], y, color='k', alpha=0.5, lw=0.5)
    ax0.set_title('Empirical data')
    
    # ax1: plot resampled data
    ax1.hist(distribution, color='k', alpha=0.5)
    ax1.axvline(np.nanmean(distribution), color='r', label='mean')
    ax1.axvline(0, color='k', linestyle='--', label='null')
    ax1.set_xlabel(f"\u0394 {variable}")
    ax1.set_ylabel('count')
    ax1.set_title('Resampled distribution')
    ax1.legend()


def _print_results(p_value, true_mean, n_iterations, conditions):
    # print p-value
    if p_value==0:
        print(f"p-value: <{1/n_iterations}")
    else:
        if p_value < 0.001:
            print(f"p-value: {p_value:.2e}")
        else:
            print(f"p-value: {p_value:.3f}")

    # print mean difference
    print(f"True mean difference: {true_mean:.2f}")
    
    # print condition comparison statement
    if true_mean > 0:
        print(f"Condition '{conditions[1]}' > '{conditions[0]}'")
    elif true_mean < 0:
        print(f"Condition '{conditions[0]}' > '{conditions[1]}'")
    else:
        print(f"Conditions '{conditions[0]}' and '{conditions[1]}' are equal")