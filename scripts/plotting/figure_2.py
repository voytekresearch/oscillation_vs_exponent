"""
Plot electrode locations on glass brain and barchart depicting number of 
electrodes in each patient.
"""

# Imports - standard
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
plt.style.use("mplstyle/default.mplstyle")


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

    # plot electrode locations - color each patient
    df['patient_color'] = df['patient'].map({p: i/len(PATIENTS) for i, p in enumerate(PATIENTS)})
    nfig = plotting.plot_markers(node_coords=df[['pos_x', 'pos_y', 'pos_z']].values, 
                        node_values=df['patient_color'], node_size=2, 
                        node_cmap='tab20', colorbar=False, display_mode='ortho', 
                        axes=ax1, annotate=False)
    nfig.annotate(size=7) # must plot with annotate=False, then set size here

    # plot bargraph of number of electrodes per patient - match colors to 'tab20' above
    colors = plt.cm.tab20.colors[:len(PATIENTS)]
    df['patient_idx'] = df['patient'].map({p: i for i, p in enumerate(PATIENTS)})
    df.groupby('patient_idx').size().plot(kind='bar', ax=ax0, color=colors, width=0.8)
    ax0.set(xlabel="patient index", ylabel="number of electrodes")
    beautify_ax(ax0)

    # save/show
    plt.savefig(f"{dir_output}/figure_2B")
    plt.savefig(f"{dir_output}/figure_2B.png")

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


if __name__ == "__main__":
    main()
