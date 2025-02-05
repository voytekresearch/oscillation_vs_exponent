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

def load_brain_meshes():
    '''
    Import brain mesh data from Fellner 2019 dataset, and return a dictionary
    containing PyVista mesh object for each hemisphere and the whole brain.

    Parameters
    ----------
    None

    Returns
    -------
    brain_mesh : dictionary
        dictionary containing PyVista mesh object for each hemisphere and the
        whole brain.

    '''

    # Load brain mesh data
    brain_mesh = dict()
    brain_mesh['right'] = load_brain_mesh('right')
    brain_mesh['left'] = load_brain_mesh('left')
    brain_mesh['both'] = load_brain_mesh('both')
    
    return brain_mesh


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

def create_electrode_mesh(elec_pos, hemisphere='both', offset=[0,0,0],
                          verbose=True):
    '''
    Create a PyVista mesh object for electrode positions.

    Parameters
    ----------
    elec_pos : Nx3 array (pos_x, pos_y, pos_z)
        electrode positions.
    hemisphere : string, optional
        hemisphere to load. The default is 'both'.
    offset : 1x3 array, optional
        offset to add to electrode positions. The default is [0,0,0].
    verbose : bool, optional
        print warning if no electrodes in given hemisphere. The default is True.

    Returns
    -------
    electrodes : pyvista PolyData object
        Electrode mesh.

    '''

    # drop electrodes in opposite hemisphere
    mask = get_hemisphere_bool(elec_pos, hemisphere)
    elec_pos = elec_pos[mask]

    # check if any electrodes are in given hemisphere
    if not elec_pos.any():
        if verbose:
            print('No electrodes in given hemisphere')
        return None
    
    # add offset to electrode positions
    elec_pos = elec_pos + offset

    # create pyvista object for electrodes
    electrodes = pv.PolyData(elec_pos)
    
    return electrodes


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


def default_camera_pos(zoomed=True):
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

    if zoomed:
        cpos = dict({
            'right'     :   [(294.5505622450821, -33.4531639749581, -1.1988451541504688),
                            (0.21547583547528149, -18.471318770726832, 15.236837161591453),
                            (0.05767181828374215, 0.03863366567995549, 0.9975877912504628)],    
            'left'      :   [(-290.36988862408833, 22.497181438511667, -16.535137645887435),
                            (0.21547583547528149, -18.471318770726832, 15.236837161591453),
                            (-0.09617116070670201, 0.08685102622515647, 0.9915684580965467)],    
            'posterior' :   [(9.742748987576016, -309.67398621660396, 56.02085566672592),
                            (-0.12530094483760684, -18.295654359106592, 9.879553722280129),
                            (0.0029693699749435204, 0.15650379629321212, 0.9876728935167577)]})

    else:
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


def get_camera_pos(hemisphere, view):
    """
    Get camera position for given hemisphere and view.

    Parameters
    ----------
    hemisphere : string
        hemisphere to load (right or left).
    view : string
        view of the brain (lateral or medial).

    Returns
    -------
    cpos : 3x3 array
        camera position for given hemisphere and view.

    """

    cpos = default_camera_pos()

    if (hemisphere == 'left') and (view == 'lateral'):
        cpos = cpos['left']
    elif (hemisphere == 'left') and (view == 'medial'):
        cpos = cpos['right']
    elif (hemisphere == 'right') and (view == 'lateral'):
        cpos = cpos['right']
    elif (hemisphere == 'right') and (view == 'medial'):
        cpos = cpos['left']

    return cpos


def get_hemisphere_bool(elec_pos, hemisphere):
    """
    Get boolean array indicating whether electrode is in given hemisphere.

    Parameters
    ----------
    elec_pos : Nx3 array (pos_x, pos_y, pos_z)
        electrode positions.
    hemisphere : string
        hemisphere to load (right or left).

    Returns
    -------
    bool_hemisphere : Nx1 array
        boolean array indicating whether electrode is in given hemisphere.

    """

    if hemisphere == 'right':
        mask = elec_pos[:, 0] > 0
    elif hemisphere == 'left':
        mask = elec_pos[:, 0] < 0
    elif hemisphere == 'both':
        mask = np.ones(elec_pos.shape[0], dtype=bool)
    else:
        raise ValueError('Invalid hemisphere. Must be "right", "left", or "both"')

    return mask