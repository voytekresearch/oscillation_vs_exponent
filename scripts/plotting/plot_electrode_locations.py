# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:40:22 2020

@author: prestonmw
"""

# Imports - general
import numpy as np
import os
from pymatreader import read_mat

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH, DATASET_PATH
from pyvista_utils import default_camera_pos, plot_electrodes

# make directory for output figures
dir_fig = f"{PROJECT_PATH}/figures/ieeg_electrode_locations"
if not os.path.exists(dir_fig):
    os.makedirs(dir_fig)
    
# Globals
X_OFFSET = 4 # lateral offset for electrode positions
PNT_SIZE = 8 # electrode plotting size 
OPACITY = 0.8 # opacity of brain mesh
BACKGROUND_COLOR = 'k'
BRAIN_COLOR = 'w'
ELEC_COLOR = 'r'

def main():
    # camera postions for each hemispheres
    cpos = default_camera_pos()

    # plot electrodes for each patient on individual brain -----------------
    
    # init
    elec_pos_all = []

    # loop through patients
    files = os.listdir(f"{DATASET_PATH}/ieeg")
    files = [f for f in files if 'words' in f]
    for fname in files:
        # load
        data_elec = read_mat(f"{DATASET_PATH}/ieeg/{fname}")
        elec_pos = data_elec['data']['elecinfo']['elecpos_bipolar']
        
        # plot each hemisphere
        for hemisphere, offset in zip(['right', 'left'], [X_OFFSET, -X_OFFSET]):        
            fname_fig = fname.replace('.mat', f"_{hemisphere}")
            plot_electrodes(elec_pos, cpos[hemisphere], hemisphere=hemisphere,
                            background_color=BACKGROUND_COLOR, brain_color=BRAIN_COLOR,
                            elec_color=ELEC_COLOR, point_size=PNT_SIZE,
                            brain_opacity=OPACITY, offset=offset, off_screen=True,
                            show=False, fname=f"{dir_fig}/{fname_fig}")
    
        # aggregate for next plot
        elec_pos_all.append(elec_pos)
    elec_pos_all = np.concatenate(elec_pos_all)

    # plot electrodes for all patient on one brain ------------------------
    
    # plot each hemisphere
    for hemisphere, offset in zip(['right', 'left'], [X_OFFSET, -X_OFFSET]):        
        fname_fig = f"all_patients_{hemisphere}"
        plot_electrodes(elec_pos_all, cpos[hemisphere], hemisphere=hemisphere,
                        background_color=BACKGROUND_COLOR, brain_color=BRAIN_COLOR,
                        elec_color=ELEC_COLOR, point_size=PNT_SIZE,
                        brain_opacity=OPACITY, offset=offset, off_screen=True,
                        show=False, fname=f"{dir_fig}/{fname_fig}")

        
if __name__ == "__main__":
    main()

