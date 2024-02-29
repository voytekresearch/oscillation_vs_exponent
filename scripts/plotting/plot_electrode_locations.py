"""
This script plots the electrode locations of each patient, and create a single 
plot for all patients.

"""

# Imports - standard
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from info import PATIENTS
from plots import plot_electrodes
    
# Plot settings
ELEC_SIZE = 24 # electrode size 
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

    # loop through hemispheres and views
    for hemisphere in ['right', 'left']:
        for view in ['lateral', 'medial']:
            # display progress
            print(f"Plotting {hemisphere} hemisphere, {view} view")

            # plot group (all patients one color)
            fname_out = f"{dir_fig}/group/{hemisphere}_hemisphere_{view}.png"
            elec_pos_all = elec_info[['pos_x', 'pos_y', 'pos_z']].values
            plot_electrodes(elec_pos_all, hemisphere, view, 
                            elec_size=ELEC_SIZE, fname_out=fname_out)
            
            # loop through each patient
            plotter = None # initialize group plotter
            for i_pat, patient in enumerate(PATIENTS):
                # get electrode positions for patient
                df_p = elec_info.loc[elec_info['patient'] == patient]
                elec_pos = df_p[['pos_x', 'pos_y', 'pos_z']].values

                # create plot for individual patient
                fname_out = f"{dir_fig}/patient/{patient}_{hemisphere}_hemisphere_{view}.png"
                plot_electrodes(elec_pos, hemisphere, view, elec_size=ELEC_SIZE, 
                                fname_out=fname_out)

                # plot each patient on a shared group plot (individually colored)
                plotter = plot_electrodes(elec_pos, hemisphere, view,
                                            elec_color=COLORS[i_pat], 
                                            elec_size=ELEC_SIZE, 
                                            plotter=plotter, 
                                            return_plotter=True)
                    
            # save group plot for given hemisphere and view
            fname_out = f"{dir_fig}/group/{hemisphere}_hemisphere_{view}_color.png"
            plotter.screenshot(fname_out)

    # combine hemispheres and views into a single figure and save
    print("Combining hemispheres and views ")
    images_0 = [plt.imread(f"{dir_fig}/group/{f}") for f in os.listdir(f"{dir_fig}/group") if 'color' in f]
    images_1 = [plt.imread(f"{dir_fig}/group/{f}") for f in os.listdir(f"{dir_fig}/group") if 'color' not in f]
    for images, tag in zip([images_0, images_1], ['_color', '']):
        image = np.concatenate(images, axis=1)
        fig, ax = plt.subplots(1,1, figsize=(20, 10))
        ax.imshow(image)
        ax.axis('off')
        fig.savefig(f"{dir_fig}/electrode_locations{tag}.png", 
                    bbox_inches='tight', dpi=300)

        
if __name__ == "__main__":
    main()
