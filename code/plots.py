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
             norm_type='log', vmin=None, vmax=None, fig=None, ax=None,
             cax=None, cbar_label=None):
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

    Returns
    -------
    fig : matplotlib figure
        Figure with plot.
    ax : matplotlib axis
        Axis with plot.
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
    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)
        return_fig = True
    else:
        return_fig = False

    # plot tfr
    ax.pcolor(time, freqs, tfr, cmap=cmap, norm=norm)

    # set labels and scale
    ax.set(yscale='log')
    plt.yticks([10, 30, 50, 70, 90], labels=['10','30','50','70','90'])

    # set title
    if not title is None:
        ax.set_title(title)

    # add colorbar
    if cax is None:
        cbar = fig.colorbar(ScalarMappable(cmap=cmap, norm=norm), ax=ax)
    else:
        cbar = fig.colorbar(ScalarMappable(cmap=cmap, norm=norm), cax=cax)
    if not cbar_label is None:
        cbar.set_label(cbar_label)

    # save fig
    if not fname_out is None:
        plt.savefig(fname_out)

    if return_fig:
        return fig, ax
    else:
        return ax

def plot_data_spatial(brain_pos, brain_tri, elec_pos, value=None,
                       cpos=None, fname_out=None, off_screen=False,
                       elec_size=8, elec_color='r', cmap=None,
                       brain_color='w', brain_opacity=1, 
                       backgrouond_color='k', backend='static',
                       return_plotter=False):
    """
    Plot data at electrode locations on brain surface. If value is not None, electrodes 
    are plotted as spheres with color determined by 'elec_color.'

    Parameters
    ----------
    brain_pos : numpy array
        Brain surface vertices.
    brain_tri : numpy array
        Brain surface triangles.
    elec_pos : numpy array
        Electrode positions.
    value : numpy array, optional
        Values to plot at electrodes. The default is None.
    cpos : list, optional
        Camera position (Pyvista). The default is None.
    fname_out : str, optional
        File name to save figure. The default is None.
    off_screen : bool, optional
        Whether to plot off screen. The default is False.
    elec_size and elec_color : int, optional
        Electrode size and color for plotting. The default is 8 and 'r'.
    cmap : str, optional
        Colormap for electrodes plotted. The default is None.
    brain_color and brain_opacity: str, optional
        Brain color and opacity for plotting. The default is 'w' and 1.
    backgrouond_color : str, optional
        Background color of plot. The default is 'k'.
    backend : str, optional
        Jupyter backend for plotting. The default is 'static'.
    return_plotter : bool, optional
        Whether to return the plotter object. The default is False.
    """
    # imports
    import pyvista as pv
    
    # create pyvista object for brain
    faces = np.hstack((3*np.ones((brain_tri.shape[0],1)), brain_tri))
    brain_mesh = pv.PolyData(brain_pos, faces.astype(int))
    
    # create figure and add brain mesh
    plotter = pv.Plotter(off_screen=off_screen)
    plotter.set_background(backgrouond_color)
    plotter.add_mesh(brain_mesh, color=brain_color, opacity=brain_opacity)
    
    # plot electrodes
    if value is None:
        plotter.add_mesh(pv.PolyData(elec_pos), point_size=elec_size, color=elec_color, \
                        render_points_as_spheres=True)
    else:
        # set colormap
        if cmap is None:
            cmap = pv.themes.DefaultTheme().cmap
        # plot electrodes
        print('plotting electrodes')
        plotter.add_mesh(pv.PolyData(elec_pos), point_size=elec_size, scalars=value, \
                        cmap=cmap, render_points_as_spheres=True)  
    # set camera position
    if cpos is not None:
        plotter.camera_position = cpos

    # save figure
    if fname_out is not None:
        plotter.screenshot(fname_out)

    # plot
    if not off_screen:
        plotter.show(jupyter_backend=backend)

    # return plotter
    if return_plotter:
        return plotter.screenshot()
    else:
        plotter.close()

        
def plot_binary_spatial(brain_pos, brain_tri, elec_pos, binary, 
                        cpos=None, fname_out=None, off_screen=False,
                        elec_size=8, elec_colors=['r','grey'], 
                        brain_color='w', brain_opacity=1, 
                        backgrouond_color='k', backend='static'):
    """
    Plot binary data at electrode locations on brain surface. 

    Parameters
    ----------
    brain_pos : numpy array
        Brain surface vertices.
    brain_tri : numpy array
        Brain surface triangles.
    elec_pos : numpy array
        Electrode positions.
    binary : numpy array of bool
        Binary values to plot at electrodes. (True = elec_color[0], False = elec_color[1])
    cpos : list, optional
        Camera position (Pyvista). The default is None.
    fname_out : str, optional
        File name to save figure. The default is None.
    off_screen : bool, optional
        Whether to plot off screen. The default is False.
    elec_size and elec_color : int, optional
        Electrode size and color for plotting. The default is 8 and 'r'.
    cmap : str, optional
        Colormap for electrodes plotted. The default is None.
    brain_color and brain_opacity: str, optional
        Brain color and opacity for plotting. The default is 'w' and 1.
    backgrouond_color : str, optional
        Background color of plot. The default is 'k'.
    backend : str, optional
        Jupyter backend for plotting. The default is 'static'.
    """
    # imports
    import pyvista as pv
    
    # create pyvista object for brain
    faces = np.hstack((3*np.ones((brain_tri.shape[0],1)), brain_tri))
    brain_mesh = pv.PolyData(brain_pos, faces.astype(int))
    
    # create figure and add brain mesh
    plotter = pv.Plotter(off_screen=off_screen)
    plotter.set_background(backgrouond_color)
    plotter.add_mesh(brain_mesh, color=brain_color, opacity=brain_opacity)
    
    # plot electrodes - color according to significance value
    chans_sig = pv.PolyData(elec_pos[binary])
    plotter.add_mesh(chans_sig, point_size=elec_size, color=elec_colors[0], \
                     render_points_as_spheres=True)
    chans_ns = pv.PolyData(elec_pos[~binary])
    plotter.add_mesh(chans_ns, point_size=elec_size, color=elec_colors[1], \
                    render_points_as_spheres=True)
    
    # set camera position
    if cpos is not None:
        plotter.camera_position = cpos

    # save figure
    if fname_out is not None:
        plotter.screenshot(fname_out)

    # plot
    if not off_screen:
        plotter.show(jupyter_backend=backend)
    else:
        plotter.close()

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


def plot_ap_params(params, time):
    """
    Plot time-series of aperiodic parameters.
    
    Parameters
    ----------
    params : FOOOFGroup object
        FOOOFGroup object containing aperiodic parameters.
    time : numpy array
        Time points corresponding to each aperiodic parameter.

    Returns
    -------
    None.
    """
    
    # imports
    from utils import extract_ap_params
    from neurodsp.plts import plot_time_series
    
    # get ap params
    offset, knee, exponent = extract_ap_params(params)
    
    # plot each ap param
    for var, variable in zip([offset, knee, exponent], 
                             ['offset', 'knee', 'exponent']):
        
        # skip knee if not fit
        if np.isnan(var).all(): continue
        
        # plot
        plot_time_series(time, var, title=variable)