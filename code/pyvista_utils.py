# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 11:09:46 2021

@author: micha
"""
# Import
import numpy as np
from os.path import join
import pyvista as pv
from pymatreader import read_mat

# Parameters
PROJECT_PATH = 'C:/Users/prestonmw/tilt_vs_fingerprint/'
DATASET_PATH = 'C:/Users/micha/datasets/SpectraltiltvsOscillations'

def load_brain_mesh(hemisphere='both'):
    '''
    Import brain mesh data from Fellner 2019 dataset, and return a PyVista
    mesh object.

    Parameters
    ----------
    hemisphere : string, optional
        hemisphere to load. The default is 'both'.

    Returns
    -------
    mesh : pyvista PolyData object
        Brain mesh.

    '''

    # Load brain mesh data
    fname_in = f"{DATASET_PATH}/scripts/additional scripts/surface_pial_{hemisphere}.mat"
    data_mesh = read_mat(fname_in, variable_names=('mesh'))
    pos = data_mesh['mesh']['pos']
    tri = data_mesh['mesh']['tri'] - 1 # matlab index begins at 1
    
    # create pyvista object for hemisphere
    faces = np.hstack((3*np.ones((tri.shape[0],1)),tri))
    mesh = pv.PolyData(pos,faces.astype(int))
    
    return mesh

def create_electrode_mesh(elec_pos, hemisphere='both', offset=None):
    '''
    Create a PyVista mesh object for electrode positions.

    Parameters
    ----------
    elec_pos : Nx3 array (pos_x, pos_y, pos_z)
        electrode positions.
    hemisphere : string, optional
        hemisphere to load. The default is 'both'.
    offset : float, optional
        offset to add to electrode positions. The default is None.

    Returns
    -------
    electrodes : pyvista PolyData object
        Electrode mesh.

    '''

    # drop electrodes in opposite hemisphere
    if hemisphere == 'right':
        xyz = elec_pos[elec_pos[:, 0]>0]
    elif hemisphere == 'left':
        xyz = elec_pos[elec_pos[:, 0]<0]
    elif hemisphere == 'both':
        xyz = elec_pos
    else:
        print('Invalid hemisphere. Must be "right", "left", or "both"')

    # check if any electrodes are in given hemisphere
    if not xyz.any():
        print('No electrodes in given hemisphere')
        return None
    
    # add offset to electrode positions
    if offset is not None:
        xyz[:, 0] = xyz[:, 0] + offset

    # create pyvista object for electrodes
    electrodes = pv.PolyData(xyz)
    
    return electrodes

 
def plot_electrodes(positions, cpos, hemisphere=None, background_color='k', 
                    brain_color='w',  elec_color='r', point_size=8, 
                    brain_opacity=0.8, offset=None, off_screen=False,
                    show=True, fname=None):
    """
    plot channel locations on brain mesh

    Parameters
    ----------
    positions : Nx3 array (pos_x, pos_y, pos_z)
        channle positions.
    cpos : 1x3 array
        campera position for pyvista plot.
    hemisphere : string, optional
        hemisphere to load. The default is None.
    background_color : string, optional
        color to plot the background. The default is 'k'.
    brain_color : string, optional
        color to plot brain. The default is 'w'.
    elec_color : string, optional
        color to plot electodes. The default is 'r'.
    point_size : float, optional
        DESCRIPTION. The default is 8.
    brain_opacity : float, optional
        opacity of brainmesh. The default is 0.8.
    offset : float, optional
        offset to add to electrode positions. The default is None.
    off_screen : bool, optional
        whether to plot off screen. The default is False

    Returns
    -------
    None.

    """
    
    # generate brain mesh
    brain_mesh = load_brain_mesh(hemisphere)
    
    # create figure and add brain mesh
    plotter = pv.Plotter(off_screen=off_screen)
    plotter.set_background(background_color)
    plotter.add_mesh(brain_mesh, color=brain_color, opacity=brain_opacity)
    
    # plot electrode locations
    electrodes = create_electrode_mesh(positions, hemisphere, offset)
    if electrodes is not None:
        plotter.add_mesh(electrodes, point_size=point_size, color=elec_color,
                        render_points_as_spheres=True)

    # set camera positions
    plotter.camera_position = cpos
    
    # save
    if fname is not None:
        plotter.screenshot(fname)

    # show
    if show:
        plotter.show(interactive=True)

    plotter.close()


def find_cpos_interactive():
    """
    Plot brain mesh as interactive pyvista plot, then return and print 
    final camera position.

    Returns
    -------
    cpos : 1x3 array
        final campera position of interactive pyvista plot

    """
    
    # Load brain mesh 
    brain = load_brain_mesh()

    # Plot brain mesh
    plotter = pv.Plotter()
    plotter.add_mesh(brain)

    # Show interactive plot and print camera position when window is closed
    cpos = plotter.show(interactive=True)
    print(plotter.camera_position)
    plotter.close()
    
    return cpos


def default_camera_pos():
    """
    Default camera positions for brain images.

    Returns
    -------
    cpos_right : 3x3 array
        default camera position for right hemisphere.
    cpos_left : 3x3 array
        default camera position for left hemisphere.
    cpos_post : 3x3 array
        default camera position for posterior brain surface.

    """
    cpos = dict({
        'right'     :   [(522.8751422878058, -8.533968557192056, 2.1119098264702054),
                        (0.21547583547528149, -18.471318770726832, 15.236837161591453),
                        (0.023688404036230154, 0.07114592740239216, 0.9971845950115104)],    
        'left'      :   [(-520.3531805339194, -38.83703309211285, 60.37721009778639),
                        (0.21547583547528149, -18.471318770726832, 15.236837161591453),
                        (0.08481614834082721, 0.038835387452391915, 0.9956395098940428)],    
        'posterior' :   [(15.14386468264713, -479.3300209752663, 261.87532035441575),
                        (0.21547583547528149, -18.471318770726832, 15.236837161591453),
                        (-0.0024849751700573297, 0.47178555248856824, 0.8817098260547291)]})
    
    return cpos
    