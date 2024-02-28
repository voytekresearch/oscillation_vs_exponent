"""
This script plots the electrode locations of each patient, and create a single 
plot for all patients.

"""

# Imports - standard
import os
import numpy as np
import pandas as pd

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from plots import plot_electrodes
    
# Plot settings
COLORS = ['#00000','#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c',
          '#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928'] # colors for each patient

def main():
    # make directory for output figures
    dir_fig = f"{PROJECT_PATH}/figures/ieeg_electrode_locations"
    if not os.path.exists(f"{dir_fig}/group"): 
        os.makedirs(f"{dir_fig}/group")
    if not os.path.exists(f"{dir_fig}/patient"): 
        os.makedirs(f"{dir_fig}/patient")
        
    # load electrode info
    fname_in = f"{PROJECT_PATH}/data/ieeg_metadata/ieeg_channel_info.csv"
    elec_info = pd.read_csv(fname_in, index_col=0)
    patients = np.unique(elec_info['patient'])

    # loop through hemispheres and views
    for hemisphere in ['right', 'left']:
        for view in ['lateral', 'medial']:
            # display progress
            print(f"Plotting {hemisphere} hemisphere, {view} view")

            # plot group (all patients one color)
            fname_out = f"{dir_fig}/group/{hemisphere}_hemisphere_{view}.png"
            elec_pos_all = elec_info[['pos_x', 'pos_y', 'pos_z']].values
            plot_electrodes(elec_pos_all, hemisphere, view, fname_out=fname_out)
            
            # loop through each patient
            plotter = None # initialize group plotter
            for i_pat, patient in enumerate(patients):
                # get electrode positions for patient
                df_p = elec_info.loc[elec_info['patient'] == patient]
                elec_pos = df_p[['pos_x', 'pos_y', 'pos_z']].values

                # create plot for individual patient
                fname_out = f"{dir_fig}/patient/{patient}_{hemisphere}_hemisphere_{view}.png"
                plot_electrodes(elec_pos, hemisphere, view, fname_out=fname_out)

                # plot each patient on a shared group plot (individually colored)
                plotter = plot_electrodes(elec_pos, hemisphere, view,
                                            elec_color=COLORS[i_pat], 
                                            plotter=plotter, 
                                            return_plotter=True)
                    
            # save group plot for given hemisphere and view
            fname_out = f"{dir_fig}/group/{hemisphere}_hemisphere_{view}_color.png"
            plotter.screenshot(fname_out)

        
if __name__ == "__main__":
    main()
