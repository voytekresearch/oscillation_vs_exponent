"""
This script plots the average power spectra for channels identified in
scripts.3_id_modulated_channels.py as being task-modulated.

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
from info import MATERIALS
from plots import plot_spectra_2conditions, beautify_ax
from settings import COLORS, FREQ_RANGE, WIDTH, BCOLORS

# settings
plt.style.use('mplstyle/default.mplstyle')


def main():

    # make directory for output figures
    dir_fig = f"{PROJECT_PATH}/figures/main_figures"
    if not os.path.exists(dir_fig): 
        os.makedirs(dir_fig)

    # load electrode info
    fname_in = f"{PROJECT_PATH}/data/ieeg_metadata/ieeg_channel_info.csv"
    df = pd.read_csv(fname_in, index_col=0).drop(columns='index')

    # load results of step 3 and merge with electrode info
    fname = f"{PROJECT_PATH}/data/results/ieeg_modulated_channels.csv"
    temp = pd.read_csv(fname, index_col=0)
    df = df.merge(temp, on=['patient', 'chan_idx'])

    # initialize figure
    fig, (axb, filler, axa, axc, axd) = plt.subplots(1, 5, constrained_layout=True,
                                           width_ratios=[2.75, 0.25, 0.75, 1, 1],
                                           figsize=(WIDTH['2col'], 
                                                    WIDTH['2col']/4))
    
    # filler cuz nilearn is weird with subplots
    filler.axis('off')
    
    # plot barchart: number of task-modulated electrodes
    x = [0, 1 , 2]
    y = [df[col].sum() / len(df) * 100 for col in ['sig_alpha', 'sig_gamma', 'sig_all']] 
    axa.bar(x, y, color=[BCOLORS['alpha'], BCOLORS['gamma'], 'grey'],
            edgecolor='black', linewidth=1, width=1)
    axa.set_xticks(x, labels=['alpha', 'gamma', 'both'])
    axa.set_ylabel('Percentage')
    axa.set_xlabel('Frequency band')
    beautify_ax(axa)
    
    # plot glass brain: electrode locations
    df_ = df[df['sig_all']]
    nfig = plotting.plot_markers(node_coords=df_[['pos_x', 'pos_y', 'pos_z']].values, 
                        node_values=df_['sig_all'], node_size=2, colorbar=False, 
                        node_cmap='binary', node_vmin=0, node_vmax=1, 
                        display_mode='ortho', axes=axb, annotate=False)
    nfig.annotate(size=7) # must plot with annotate=False, then set size here

    # plot spectra: group mean for word and face blocks
    plot_group_spectra(df, [axc, axd])

    # set titles
    axa.set_title("Task-modulated electrodes\nby frequency band")
    axb.set_title("                   Task-modulated electrode locations")
    axc.set_title("\nword block")
    axd.set_title("\nface block")
    fig.text(0.8, 0.97, "Mean power spectra", ha='center', va='center',
             fontsize=7)

    # save
    plt.savefig(f"{dir_fig}/figure_4")
    plt.savefig(f"{dir_fig}/figure_4.png")
    plt.close()


def plot_group_spectra(df, axes):

    for ax, material, color in zip(axes, MATERIALS, ['brown', 'blue']):
        # load data
        fname = f"{PROJECT_PATH}/data/ieeg_spectral_results/psd_{material}_hit_XXXstim.npz"
        data_pre = np.load(fname.replace("XXX", "pre"))
        data_post = np.load(fname.replace("XXX", "post"))
        psd_pre = data_pre['spectra'][df[f"sig_all"]]
        psd_post = data_post['spectra'][df[f"sig_all"]]
        freq = data_pre['freq']

        # plot
        # title = f"{material[0].upper()}{material[1:-1]} encoding"
        colors = [COLORS[f'light_{color}'], COLORS[color]]
        f_mask = np.logical_and(freq>FREQ_RANGE[0], freq<FREQ_RANGE[1])
        plot_spectra_2conditions(psd_pre[:, f_mask], psd_post[:, f_mask], 
                                freq[f_mask], shade_sem=True, ax=ax, 
                                color=colors)
        
        # beautify
        ax.grid(False)
        beautify_ax(ax)

        
if __name__ == "__main__":
    main()
