# -*- coding: utf-8 -*-
"""
Plotting functions
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize, LogNorm, CenteredNorm, TwoSlopeNorm

# set plotting parameers
mpl.rcParams['figure.facecolor'] = 'w'
mpl.rcParams['axes.facecolor'] = 'w'
mpl.rcParams['figure.titlesize'] = 18
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['font.size'] = 10

def plot_tfr(time, freqs, tfr, fname_out=None, title=None,
             norm_type=None, vmin=None, vmax=None):
    """
    Plot time-frequency representation (TFR)
    """

    # Define a color map and normalization of values
    if vmin is None:
        vmin = np.nanmin(tfr)
    if vmax is None:
        vmax = np.nanmax(tfr)

    if norm_type == 'linear':
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = 'hot'
    elif norm_type == 'log':
        norm = LogNorm(vmin=vmin, vmax=vmax)
        cmap = 'hot'
    elif norm_type == 'centered':
        norm = CenteredNorm(vcenter=0)
        cmap = 'coolwarm'
    elif norm_type == 'two_slope':
        norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
        cmap = 'coolwarm'
    else:
        print("norm_type must be 'linear', 'log', 'centered', or 'two_slope'")
    
    # plot
    fig, ax = plt.subplots(constrained_layout=True)
    ax.pcolor(time, freqs, tfr, cmap=cmap, norm=norm)

    # set labels and scale
    ax.set(yscale='log')
    plt.yticks([10, 30, 50, 70, 90], labels=['10','30','50','70','90'])

    # set title
    if not title is None:
        ax.set_title(title)

    # save fig
    if not fname_out is None:
        plt.savefig(fname_out)

    # show fig
    plt.show()

    return fig, ax


