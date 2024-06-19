# -*- coding: utf-8 -*-
"""
Plotting functions
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, CenteredNorm, TwoSlopeNorm


def set_custom_colors(keyword=None):
    """
    Add custom colors to matplotlib.
    """

    import matplotlib as mpl
    from cycler import cycler

    if keyword is None:
        mpl.rcParams['axes.prop_cycle'] = cycler(color=[
            np.array([166,97,26]) / 255, # brown
            np.array([1,133,113]) / 255, # blue
        ])
    
    elif keyword == 'browns':
        mpl.rcParams['axes.prop_cycle'] = cycler(color=[
            np.array([223,194,125]) / 255, # light brown
            np.array([166,97,26]) / 255, # brown
        ])

    elif keyword == 'blues':
        mpl.rcParams['axes.prop_cycle'] = cycler(color=[
            np.array([128,205,193]) / 255, # light blue
            np.array([1,133,113]) / 255, # blue
        ])
                
    elif keyword == 'all':
        mpl.rcParams['axes.prop_cycle'] = cycler(color=[
            np.array([223,194,125]) / 255, # light brown
            np.array([166,97,26]) / 255, # brown
            np.array([128,205,193]) / 255, # light blue
            np.array([1,133,113]) / 255, # blue
        ])

    elif keyword == 'default':
        mpl.rcParams['axes.prop_cycle'] = plt.rcParamsDefault['axes.prop_cycle']
        
    else:
        raise ValueError('Keyword not recognized. Options are "default", \
                         "browns", "blues", "all", or "default".')


def plot_evoked_tfr(tfr, freq, time, **kwargs):
    """
    Plot evoked time-frequency representation (TFR). At each frequnecy, the
    power is z-scored and baseline subtracted before plotting.
    """

    # imports
    from tfr_utils import zscore_tfr, subtract_baseline

    # z-score and subtract baseline for visualization of evoked power
    tfr  = zscore_tfr(tfr)
    tfr = subtract_baseline(tfr, time)

    # plot
    plot_tfr(time, freq, tfr, norm_type='centered', **kwargs)


def plot_tfr(time, freqs, tfr, fname_out=None, title=None,
             norm_type='log', vmin=None, vmax=None, fig=None, ax=None,
             cax=None, cbar_label=None, annotate_zero=False, log_yscale=False):
    """
    Plot time-frequency representation (TFR)

    Parameters
    ----------
    time : 1D array
        Time vector.
    freqs : 1D array
        Frequency vector.
    tfr : 2D array
        Time-frequency representation of power (spectrogram).
    fname_out : str, optional
        File name to save figure. The default is None.
    title : str, optional
        Title of plot. The default is None.
    norm_type : str, optional
        Type of normalization for color scale. Options are 'linear', 'log',
        'centered', and 'two_slope'. The default is 'log'.
    vmin, vmax : float, optional
        Minimum/maximum value for color scale. The default is None, which
        sets the min/max to the min/max of the TFR.
    fig : matplotlib figure, optional
        Figure to plot on. The default is None, which creates a new figure.
    ax : matplotlib axis, optional
        Axis to plot on. The default is None, which creates a new axis.
    cax : matplotlib axis, optional
        Axis to plot colorbar on. The default is None.
    cbar_label : str, optional
        Label for colorbar. The default is None.
    annotate_zero : bool, optional
        Whether to annotate zero on the time axis. The default is False.
    log_yscale : bool, optional
        Whether to use a log scale for the y-axis. The default is False.

    Returns
    -------
    None.
    """

    # imports
    from matplotlib.cm import ScalarMappable

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
    
    # create figure
    if (ax is None) & (fig is None):
        fig, ax = plt.subplots(constrained_layout=True)
    elif (ax is None) | (fig is None):
        raise ValueError('Both fig and ax must be provided if one is provided.')

    # plot tfr
    ax.pcolor(time, freqs, tfr, cmap=cmap, norm=norm)

    # set labels and scale
    if log_yscale is True:
        ax.set(yscale='log')
        ax.set_yticks([10, 100])
        ax.set_yticklabels(['10','100'])

    # set title
    if not title is None:
        ax.set_title(title)

    # label axes
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')

    # add colorbar
    if cax is None:
        cbar = fig.colorbar(ScalarMappable(cmap=cmap, norm=norm), ax=ax)
    else:
        cbar = fig.colorbar(ScalarMappable(cmap=cmap, norm=norm), cax=cax)
    if not cbar_label is None:
        cbar.set_label(cbar_label)

    # annotate zero
    if annotate_zero:
        ax.axvline(0, color='k', linestyle='--', linewidth=2)

    # add grid
    ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.8)

    # save fig
    if not fname_out is None:
        plt.savefig(fname_out)

 
def plot_electrodes(positions, hemisphere='right', view='lateral', plotter=None, 
                    elec_color='r', elec_size=12, elec_offset=[0,0,0],
                    brain_color='w', brain_opacity=0.7, return_plotter=False,
                    fname_out=None):
    """
    plot channel locations on brain mesh

    Parameters
    ----------
    positions : Nx3 array (pos_x, pos_y, pos_z)
        channle positions.
    hemisphere : string, optional
        hemisphere to plot ('right' or 'left'). The default is 'right'.
    view : string, optional
        view of hemisphere ('lateral' or 'medial'). The default is 'lateral'.
    plotter : pyvista plotter object, optional
        Pyvista plotter object to add data to. If None, a new plotter is created. 
        The default is None.
    elec_color : string, optional
        color to plot electodes. The default is 'r'.
    elec_size : float, optional
        DESCRIPTION. The default is 8.
    elec_offset : list, optional
        offset to add to electrode positions [x,y,z]. The default is [0,0,0].
    brain_color : string, optional
        color to plot brain. The default is 'w'.
    brain_opacity : float, optional
        opacity of brainmesh. The default is 0.75.
    return_plotter : bool, optional
        whether to return plotter object. If False, plotter is shown. The
        default is False.
    fname_out : str, optional
        file name for figure. If None, figure is not saved. The default is None.

    Returns
    -------
    plotter : pyvista plotter object
        plotter object with brain and electrode locations. Only returned if
        return_plotter is True.

    """

    # imports
    import pyvista as pv
    from pyvista_utils import (load_brain_mesh, create_electrode_mesh, 
                               get_camera_pos)
    
    # create plotter object
    if plotter is None:
        plotter = pv.Plotter(off_screen=True)
        plotter.set_background('w')

        # create brain mesh and add to plotter
        brain_mesh = load_brain_mesh(hemisphere)
        plotter.add_mesh(brain_mesh, color=brain_color, opacity=brain_opacity)
    
    # plot electrode locations
    electrodes = create_electrode_mesh(positions, hemisphere, elec_offset, 
                                       verbose=False)
    if electrodes is not None:
        plotter.add_mesh(electrodes, point_size=elec_size, color=elec_color,
                        render_points_as_spheres=True)

    # set camera position
    cpos = get_camera_pos(hemisphere, view)
    plotter.camera_position = cpos
    
    # save
    if fname_out is not None:
        plotter.screenshot(fname_out)

    # return plotter
    if not return_plotter:
        if fname_out is None:
            plotter.show(jupyter_backend='static')
    else:
        return plotter


def plot_data_spatial(values, positions, hemisphere='right', view='lateral', 
                      plotter=None, cmap='viridis', clim=None, cbar_label='', 
                      plot_cbar=True, elec_size=12, elec_offset=[0,0,0], 
                      brain_color='w', brain_opacity=0.7, 
                      return_plotter=False, fname_out=None):
    """
    Plot data at electrode locations on brain surface. 

    Parameters
    ----------
    values : numpy array
        Values to plot at electrodes.
    positions : numpy array
        Electrode positions.
    hemisphere : str, optional
        Hemisphere to plot. The default is 'right'.
    view : str, optional
        View of hemisphere ('lateral' or 'medial'). The default is 'lateral'.
    plotter : pyvista plotter object, optional
        Pyvista plotter object to add data to. If None, a new plotter is created. 
        The default is None.
    cmap : str, optional
        Colormap for electrodes plotted. The default is 'viridis'.
    clim : list, optional
        Color limits for colormap. The default is None.
    cbar_label : str, optional
        Label for colorbar. The default is ''.
    plot_cbar : bool, optional
        Whether to plot colorbar. The default is True.
    elec_size : int, optional
        Size of electrodes. The default is 8.
    elec_offset : list, optional
        Offset to add to electrode positions [x,y,z]. The default is [0,0,0].
    brain_color : str, optional
        Color of brain surface. The default is 'w'.
    brain_opacity : float, optional
        Opacity of brain surface. The default is 0.75.       
    return_plotter : bool, optional
        Whether to return plotter object. If False, plotter is shown. The
        default is False.
    fname_out : str, optional
        File name to save figure. The default is None.
    """

    # imports
    import pyvista as pv
    from pyvista_utils import (get_hemisphere_bool, load_brain_mesh, 
                               create_electrode_mesh, get_camera_pos)
    
    # filter values and positions for hemisphere
    mask = get_hemisphere_bool(positions, hemisphere)
    positions = positions[mask]
    values = values[mask]
    
    # create plotter object
    if plotter is None:
        plotter = pv.Plotter(off_screen=True)
        plotter.set_background('w')

        # create brain mesh and add to plotter
        brain_mesh = load_brain_mesh(hemisphere)
        plotter.add_mesh(brain_mesh, color=brain_color, opacity=brain_opacity)
    
    # create electrode mesh and add to plotter
    elec_mesh = create_electrode_mesh(positions, hemisphere, elec_offset, 
                                      verbose=False)
    
    if plot_cbar:
        scalar_bar_args = {'title' : cbar_label, 
                        'title_font_size' : 24,
                        'color' : 'k'}
        if clim is None:
            clim = [np.nanmin(values), np.nanmax(values)]
        plotter.add_mesh(elec_mesh, point_size=elec_size, scalars=values, 
                        cmap=cmap, clim=clim, render_points_as_spheres=True, 
                        scalar_bar_args=scalar_bar_args)
    else:
        plotter.add_mesh(elec_mesh, point_size=elec_size, scalars=values, 
                        cmap=cmap, clim=clim, render_points_as_spheres=True,
                        show_scalar_bar=False)
        
    # set camera position
    cpos = get_camera_pos(hemisphere, view)
    plotter.camera_position = cpos

    # save figure
    if fname_out is not None:
        plotter.screenshot(fname_out)

    # return plotter
    if not return_plotter:
        if fname_out is None:
            plotter.show(jupyter_backend='static')
    else:
        return plotter


def find_cpos_interactive():
    """
    Plot brain mesh as interactive pyvista plot, then return and print 
    final camera position.
    NOTE: must be in base director of Fellner data repository.

    Returns
    -------
    cpos : 1x3 array
        final campera position of interactive pyvista plot

    """
    # imports
    import pyvista as pv
    from scipy.io import loadmat

    # Load brain mesh data
    fname_in = r"C:\Users\micha\datasets\SpectraltiltvsOscillations\Scripts\additional scripts\surface_pial_both.mat"
    data_mesh = loadmat(fname_in, variable_names=('mesh'))
    pos = data_mesh['mesh']['pos'][0][0]
    tri = data_mesh['mesh']['tri'][0][0] - 1 # matlab index begins at 1
    
    # create pyvista object for hemisphere
    faces = np.hstack((3*np.ones((tri.shape[0],1)),tri))
    brain = pv.PolyData(pos,faces.astype(int))
    
    # create figure and add brain mesh
    plotter = pv.Plotter()
    plotter.add_mesh(brain)
    
    # show
    cpos = plotter.show(interactive=True)
    print(plotter.camera_position)
    plotter.close()
    
    return cpos


def plot_ap_params(params, time, single_plot=True):
    """
    Plot time-series of aperiodic parameters.
    
    Parameters
    ----------
    params : SpectralGroupGroup object
        SpectralGroupGroup object containing aperiodic parameters.
    time : numpy array
        Time points corresponding to each aperiodic parameter.
    single_plot : bool, optional
        Whether to plot all aperiodic parameters on a single plot.
        The default is True.

    Returns
    -------
    None.
    """
    
    # imports
    from specparam_utils import extract_ap_params
    from neurodsp.plts import plot_time_series
    
    # get ap params
    offset, knee, exponent = extract_ap_params(params)

    # plot each ap param separately
    if not single_plot:
        # create figure
        fig, axes = plt.subplots(3, 1, figsize=(10, 7.5))

        # plot each ap param
        for var, variable, ax in zip([offset, knee, exponent], 
                                    ['offset', 'knee', 'exponent'],
                                    axes):
            
            # skip knee if not fit
            if np.isnan(var).all(): continue
            
            # plot
            plot_time_series(time, var, title=variable, ax=ax)
            ax.set_ylabel(variable)

    # plot all ap params on a single plot
    else:
        # zscore params
        offset = (offset - np.mean(offset)) / np.std(offset)
        knee = (knee - np.mean(knee)) / np.std(knee)
        exponent = (exponent - np.mean(exponent)) / np.std(exponent)    

        # join signals as 2D array
        signals = np.vstack((offset, knee, exponent))

        # plot
        plot_time_series(time, signals, labels=['offset', 'knee', 'exponent'],
                         title='Aperiodic parameters')

        # label
        plt.ylabel('z-score')



def plot_spectra_2conditions(psd_pre, psd_post, freq, ax=None, shade_sem=True,
                             color=['grey','k'], labels=['baseline','encoding'],
                             y_units='\u03BCV\u00b2/Hz', title=None, fname=None):
    
    """
    Plot mean spectra for two conditions, with optional shading of SEM.

    Parameters
    ----------
    psd_pre : 2d array
        PSD values for baseline condition.
    psd_post : 2d array
        PSD values for encoding condition.
    freq : 1d array
        Frequency values corresponding to PSD values.
    ax : matplotlib axis, optional
        Axis to plot on. The default is None.
    shade_sem : bool, optional
        Whether to shade SEM. The default is True.
    color : list, optional
        Colors for each condition. The default is ['grey','k'].
    labels : list, optional
        Labels for each condition. The default is ['baseline','encoding'].
    y_units : str, optional
        Units for y-axis. The default is '\u03BCV\u00b2/Hz' (microvolts).
    title : str, optional
        Title for plot. The default is None.
    fname : str, optional
        File name to save figure. The default is None.

    Returns
    -------
    None.
    """

    # check axis
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=[6,4])

    # check psds are 2d
    if not (psd_pre.ndim == 2 and psd_post.ndim == 2):
        raise ValueError('PSDs must be 2d arrays.')
    
    # remove rows containing all nans
    psd_pre = psd_pre[~np.isnan(psd_pre).all(axis=1)]
    psd_post = psd_post[~np.isnan(psd_post).all(axis=1)]

    # plot mean spectra for each condition
    ax.loglog(freq, np.mean(psd_pre, axis=0), color=color[0], label=labels[0])
    ax.loglog(freq, np.mean(psd_post, axis=0), color=color[1], label=labels[1])
    
    # shade between SEM of spectra for each condition
    if shade_sem:
        ax.fill_between(freq, np.mean(psd_pre, axis=0) - (np.std(psd_pre, axis=0)/np.sqrt(psd_pre.shape[0])),
                        np.mean(psd_pre, axis=0) + (np.std(psd_pre, axis=0)/np.sqrt(psd_pre.shape[0])), 
                        color=color[0], alpha=0.5)
        ax.fill_between(freq, np.mean(psd_post, axis=0) - (np.std(psd_post, axis=0)/np.sqrt(psd_post.shape[0])),
                        np.mean(psd_post, axis=0) + (np.std(psd_post, axis=0)/np.sqrt(psd_post.shape[0])),
                        color=color[1], alpha=0.5)

    # set axes ticks and labels
    ax.set_ylabel(f'power ({y_units})')
    ax.set_xlabel('frequency (Hz)')
    ax.set_xticks([10,100])
    ax.set_xticklabels(["10", "100"])

    # add legend
    ax.legend(labels)

    # add title
    if title is None:
        ax.set_title('Power spectra')
    else:
        ax.set_title(title)

    # add grid
    ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5)
    
    # return
    if fname is not None:
        plt.savefig(fname)


def plot_psd_diff(freq, psd_diff, title=None, fname_out=None,
                  plot_each=False, plot_mean=True, shade_sem=True,
                  y_units='\u03BCV\u00b2/Hz', ax=None):
    """ 
    Plot spectra (or change in spectral power) in semi-log space.
    The mean spectrum is plotted in black, and the individual spectra are plotted in grey.
    A horizontal line at power=0 is also plotted.

    Parameters
    ----------
    freq : array
        Frequency values.
    psd_diff : array
        Spectral power values (difference in log power between 2 spectra).
    title : str, optional
        Title of plot. The default is None.
    fname_out : str, optional
        Path to save figure to. If None, figure is not saved.
        The default is None.
    plot_each : bool, optional
        Whether to plot each individual spectrum. The default is False.
    plot_mean : bool, optional
        Whether to plot the mean spectrum. The default is True.
    shade_sem : bool, optional
        Whether to shade the standard error of the mean. The default is True.
    y_units : str, optional
        Units for y-axis. The default is '\u03BCV\u00b2/Hz' (microvolts).
    ax : matplotlib axis, optional
        Axis to plot on. If None, a new figure is created.

    Returns
    -------
    None.
    
    """
    # create figure
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6,4))

    # plot psd
    if plot_each:
        ax.plot(freq, psd_diff.T, color='grey')

    # plot mean
    if plot_mean:
        ax.plot(freq, np.nanmean(psd_diff, axis=0), color='k', linewidth=3,
                label="mean")

    # shade sem
    if shade_sem:
        mean = np.nanmean(psd_diff, axis=0)
        sem = np.nanstd(psd_diff, axis=0) / np.sqrt(psd_diff.shape[0])
        ax.fill_between(freq, mean - sem, mean + sem,
                        color='k', alpha=0.2, label="SEM")

    # legend
    try:
        ax.legend()
    except:
        pass

    # scale x-axis logarithmically
    ax.set(xscale="log")

    # set axes ticks and labels
    ax.set_ylabel(f'log power ({y_units})')
    ax.set_xlabel('frequency (Hz)')
    ax.set_xticks([10,100])
    ax.set_xticklabels(["10", "100"])

    # title
    if title is None:
        ax.set_title(f"Difference in power")
    else:
        ax.set_title(title)

    # annotate power=0
    ax.axhline(0, color='r', linestyle='--', linewidth=3)

    # add grid
    ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5)

    # save
    if not fname_out is None:
        plt.savefig(fname_out)

