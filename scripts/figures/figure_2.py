"""
Plot electrode locations on glass brain and barchart depicting number of 
electrodes in each patient.
"""

# Imports - standard
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from nilearn import plotting

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from info import PATIENTS
from settings import WIDTH
from plots import beautify_ax
from utils import get_start_time, print_time_elapsed

# settings
plt.style.use("mplstyle/nature_neuro.mplstyle")


def main():

    # display progress
    t_start = get_start_time()

    # identify / create directories
    dir_output = f"{PROJECT_PATH}/figures/main_figures"
    if not os.path.exists(dir_output): 
        os.makedirs(f"{dir_output}")

    # load electrode info
    fname_in = f"{PROJECT_PATH}/data/ieeg_metadata/ieeg_channel_info.csv"
    df = pd.read_csv(fname_in, index_col=0).drop(columns='index')

    # initialize figure
    _, (ax0, ax1) = plt.subplots(1, 2, figsize=(WIDTH['2col'], WIDTH['2col']/4), 
                               width_ratios=[1, 3], constrained_layout=True)

    # plot bargraph of number of electrodes per patient
    colors = plt.cm.tab20.colors[:len(PATIENTS)]
    df['patient_idx'] = df['patient'].map({p: i + 1 for i, p in enumerate(PATIENTS)})
    df.groupby('patient_idx').size().plot(kind='bar', ax=ax0, color=colors, width=0.8)
    ax0.set(xlabel="patient index", ylabel="number of electrodes")
    beautify_ax(ax0)

    # plot electrode locations - match colors to bar graph
    coords = df[['pos_x', 'pos_y', 'pos_z']].values
    nfig = plotting.plot_markers(axes=ax1, node_coords=coords, node_size=1, 
                          node_values=df['patient_idx'],
                            display_mode='ortho', colorbar=False,
                            annotate=False, alpha=1,
                            node_cmap=mpl.colors.ListedColormap(colors))
    
    # remove gyri/sulci lines
    for xyz in nfig.axes:
        for axx in nfig.axes[xyz].ax.get_children():
            if type(axx) == mpl.patches.PathPatch:
                if axx.get_edgecolor()[0] == 0.6509803921568628:
                    axx.remove()
    nfig.annotate(size=7) # must plot with annotate=False, then set size here

    # save/show
    plt.savefig(f"{dir_output}/figure_2bc")
    plt.savefig(f"{dir_output}/figure_2bc.png")

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


if __name__ == "__main__":
    main()
