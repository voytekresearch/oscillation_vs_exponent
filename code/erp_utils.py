# -*- coding: utf-8 -*-
"""
ERP utility functions
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt

def subtract_baseline(signal, time, t_baseline):
    """Subtracts the mean of the signal in a given time interval from the signal.

    Parameters
    ----------
    signal : array
        Signal array to subtract mean from
    time : array
        Array with the corresponding time points of the signal array
    t_baseline : array
        Array with the starting and ending time points for the time interval

    Returns
    -------
    signal : array
        Signal array with mean of the given time interval subtracted
    """

    baseline = np.logical_and(time>=t_baseline[0], time<=t_baseline[1])
    signal = signal - np.mean(signal[baseline])

    return signal


def compute_erp(signal, time, t_baseline):
    """Compute the event-related potential (ERP) of a signal.

    Parameters
    ----------
    signal : ndarray
        Input signal with shape (n_trials, n_channels, n_samples).
    time : ndarray
        Timepoints associated with the signal.
    t_baseline : list
        Interval of timepoints used to calculate the baseline of each trial.

    Returns
    -------
    erp : ndarray
        Event-related potential with shape (n_channels, n_samples).
    """

    # subtract baseline
    signal_norm = np.zeros_like(signal)
    for i_trial in range(signal.shape[0]):
        for i_channel in range(signal.shape[1]):
            signal_norm[i_trial][i_channel] = subtract_baseline(signal[i_trial][i_channel], time, t_baseline)

    # compute ERP (channel-wise)
    erp = np.nanmean(signal_norm, axis=0)

    return erp


def plot_erp(erp, time, ax=None, x_units='s', y_units='\u03BCV', 
    annotate_times=[0], legend_labels=None):
    """Plots a voltage versus time graph of an evoked response (ERP).

    Parameters
    ----------
    erp : array_like
        Voltage vector of the ERP.
    time : array_like
        Time vector of the ERP.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on (default is None).
    x_units : str, optional
        Units for the x-axis (default is 's').
    y_units : str, optional
        Units for the y-axis (default is 'μV').
    annotate_tims : list, optional
        List of times to annotate (default is [0]).
    legend_labels : list, optional
        List of labels for the legend (default is None).

    Returns
    -------
    None.
    """

    # create figure
    if ax is None:
        fig, ax = plt.subplots()

    # one ERP only
    if np.ndim(erp) == 1:
        # plot
        ax.plot(time, erp, color='k')

    # multiple ERPs
    elif np.ndim(erp) ==2:
        # plot
        ax.plot(time, erp.T, color='k', alpha=0.2)

        # label
        if not legend_labels is None:
            ax.legend(legend_labels)

    else:
        raise ValueError('ERP must be a 1D or 2D array')
    
    # label
    ax.set(xlabel=f'time ({x_units})', ylabel=f'voltage ({y_units})')
    ax.set_title('Evoked Response')

    # annotate
    for i_time in annotate_times:
        ax.axvline(i_time, color='k', linestyle='--')
    
    
def plot_event_traces(event_traces, time, ax=None, annotate_time=0,
    plot_mean=True, plot_std=True, plot_individual=False,
    x_units='s', y_units='\u03BCV'):

    """Plots event traces and related statistics.

    Parameters
    ----------
    event_traces : array_like
        An array of event traces
    time : array_like
        An array of time points corresponding to the event traces
    ax : matplotlib.axes.Axes, optional
        The axis to plot on. The default is None
    annotate_time : int, optional
        The time point at which to draw an annotation line. The default is 0
    plot_mean : bool, optional
        Whether or not to plot the mean trace. The default is True
    plot_std : bool, optional
        Whether or not to plot the standard deviation. The default is True
    plot_individual : bool, optional
        Whether or not to plot individual traces. The default is False
    x_units : str, optional
        The string to use in the x-axis label. The default is 's'
    y_units : str, optional
        The string to use in the y-axis label. The default is 'µV'

    Returns
    -------
    None.
    """

    # create figure
    fig, ax = plt.subplots()

    # plot individual traces
    if plot_individual:
        ax.plot(time, event_traces.T, color='grey', alpha=0.5)

    # plot mean
    et_mean = np.mean(event_traces, axis=0)
    if plot_mean:
        ax.plot(time, et_mean, color='k')

    # plot standard deviation
    if plot_std:
        et_std = np.std(event_traces, axis=0)
        ax.fill_between(time, et_mean-et_std, et_mean+et_std, color='grey', alpha=0.5)

    # label
    ax.set(xlabel=f'time ({x_units})', ylabel=f'voltage ({y_units})')
    ax.set_title('Evoked Response')

    # annotate
    if not annotate_time is None:
        ax.axvline(annotate_time, color='k', linestyle='--')
