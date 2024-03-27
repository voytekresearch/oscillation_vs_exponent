"""
This script plots the results of scripts.3_id_modulated_channels.py, which
identifies channels wth significant modulation of alpha and/or gamma power.

"""

# Imports - standard
import os
import pandas as pd
import matplotlib.pyplot as plt

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from plots import plot_electrodes
from utils import combine_images
    
# Plot settings
ELEC_SIZE = 24 # electrode size 
plt.style.use('mplstyle/default.mplstyle')


def main():

    # load results of step 3
    fname = f"{PROJECT_PATH}/data/results/ieeg_modulated_channels.csv"
    results = pd.read_csv(fname, index_col=0)

    # load electrode coordinate info and merge with results
    fname_in = f"{PROJECT_PATH}/data/ieeg_metadata/ieeg_channel_info.csv"
    elec_info = pd.read_csv(fname_in, index_col=0)
    results = results.merge(elec_info, on=['patient','chan_idx'])
    elec_pos = results[['pos_x', 'pos_y', 'pos_z']].values

    # loop through frequency bands
    for band in ['alpha', 'gamma', 'all', 'any']:
        # display progress
        print(f"Plotting {band} modulated channels...")

        # make directory for output figures
        dir_fig = f"{PROJECT_PATH}/figures/modulated_chans/{band}"
        if not os.path.exists(f"{dir_fig}"): 
            os.makedirs(f"{dir_fig}")

        # loop through hemispheres and views
        for hemisphere in ['right', 'left']:
            for view in ['lateral', 'medial']:
                # display progress
                print(f"\tPlotting {hemisphere} hemisphere, {view} view")

                # plot electrodes
                plotter = plot_electrodes(elec_pos[results[f'sig_{band}']], hemisphere, 
                                        view, elec_color='r', elec_size=ELEC_SIZE,
                                        return_plotter=True)
                plot_electrodes(elec_pos[~results[f'sig_{band}']], hemisphere, view, 
                                elec_color='grey', elec_size=ELEC_SIZE,
                                plotter=plotter)
                
                # save group plot for given hemisphere and view
                fname_out = f"{dir_fig}/{hemisphere}_hemisphere_{view}.png"
                plotter.screenshot(fname_out)

        # combine hemispheres and views into a single figure and save
        print("Combining hemispheres and views ")
        fname_out = f"{PROJECT_PATH}/figures/modulated_chans/{band}_modulated_chans.png"
        combine_images(dir_fig, fname_out)

        
if __name__ == "__main__":
    main()
