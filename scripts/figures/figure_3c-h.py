"""
This script reproduces figure 3: Task-modulated electrodes. The results of 
scripts.3_id_modulated_channels.py are plotted.
A) Bar chart depicting percentage of task-modulated electrodes
B) Glass brain plot showing locations of task-modulated electrodes
C) Group mean power spectra for word block
D) Group mean power spectra for face block

"""

# Imports - standard
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from nilearn import plotting

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from plots import plot_spectra_2conditions, beautify_ax, join_two_figures
from settings import COLORS, FREQ_RANGE, WIDTH, BCOLORS, BANDS

# settings
plt.style.use('mplstyle/nature_neuro.mplstyle')


def main():

    # make directory for output figures
    dir_fig = f"{PROJECT_PATH}/figures/main_figures"
    if not os.path.exists(dir_fig): 
        os.makedirs(dir_fig)

    # load electrode info
    fname_in = f"{PROJECT_PATH}/data/ieeg_metadata/ieeg_channel_info.csv"
    df = pd.read_csv(fname_in, index_col=0).drop(columns='index')

    # load results of step 3 and merge with electrode info
    fname = f"{PROJECT_PATH}/data/results/band_power_statistics.csv"
    temp = pd.read_csv(fname, index_col=0)
    temp = temp.loc[temp['memory']=='hit']
    df_w = df.merge(temp.loc[temp['material']=='words'], on=['patient', 'chan_idx'])
    df_f = df.merge(temp.loc[temp['material']=='faces'], on=['patient', 'chan_idx'])
    for df in [df_w, df_f]: # compute joint significance
        df['sig_all'] = df[[f'{band}_sig' for band in BANDS]].all(axis=1)
        df['sig_any'] = df[[f'{band}_sig' for band in BANDS]].any(axis=1)

    # create figure and gridspec
    fig = plt.figure(figsize=(WIDTH['2col'], WIDTH['2col']/2), 
                     constrained_layout=True)
    spec = gridspec.GridSpec(figure=fig, ncols=3, nrows=5,
                            width_ratios=[1, 3, 1.5], 
                            height_ratios=[1, 1, 0.1, 1, 1])
    ax_u0 = fig.add_subplot(spec[0:2,0])
    ax_u1u = fig.add_subplot(spec[0,1])
    ax_u1l = fig.add_subplot(spec[1,1])
    ax_u2 = fig.add_subplot(spec[0:2,2])
    ax_header = fig.add_subplot(spec[2,:])
    ax_l0 = fig.add_subplot(spec[3:,0])
    ax_l1u = fig.add_subplot(spec[3,1])
    ax_l1l = fig.add_subplot(spec[4,1])
    ax_l2 = fig.add_subplot(spec[3:,2])

    # shift subplot spaceing (nilearn plot including inexplicable whitespace)
    for ax in [ax_u1u, ax_u1l]:
        boxb = ax.get_position()
        ax.set_position([boxb.x0-0.02, boxb.y0+0.05, boxb.width, boxb.height])
    for ax in [ax_l1u, ax_l1l]:
        boxb = ax.get_position()
        ax.set_position([boxb.x0-0.02, boxb.y0-0.05, boxb.width, boxb.height])

    # plot barchart: number of task-modulated electrodes
    for df, ax in zip([df_w, df_f], [ax_u0, ax_l0]):
        x = [0, 1, 2]
        y = [df[col].sum() / len(df) * 100 for col in ['alpha_sig', 'gamma_sig', 'sig_all']] 
        ax.bar(x, y, color=[BCOLORS['alpha'], BCOLORS['gamma'], 'k'],
                edgecolor='black', linewidth=1, width=1)
        ax.set_xticks(x, labels=['alpha', 'gamma', 'both'])
        ax.set_ylabel('percentage (%)')
        ax.set_xlabel('frequency band')
        beautify_ax(ax)
    
    # plot glass brain: electrode locations
    for df, axes in zip([df_w, df_f], [[ax_u1u, ax_u1l], [ax_l1u, ax_l1l]]):
        for ax, band in zip(axes, ['alpha', 'gamma']):
            # shift subplot spaceing (nilearn plot including inexplicable whitespace)
            coords = df.loc[df[f'{band}_sig'], ['pos_x', 'pos_y', 'pos_z']].values
            nfig = plotting.plot_markers(axes=ax, node_coords=coords, 
                                        node_values=np.ones(len(coords)), 
                                        node_cmap=ListedColormap([BCOLORS[band]]),
                                        display_mode='ortho', colorbar=False, 
                                        annotate=False, node_size=0.25, 
                                        node_kwargs={'alpha' : 1})
            coords = df.loc[df[f'sig_all'], ['pos_x', 'pos_y', 'pos_z']].values
            nfig.add_markers(marker_coords=coords, marker_size=0.25, 
                             marker_color='k', alpha=1)
            nfig.annotate(size=7) # must plot with annotate=False, then set size here

            # remove gyri/sulci lines
            for xyz in nfig.axes:
                for axx in nfig.axes[xyz].ax.get_children():
                    if type(axx) == mpl.patches.PathPatch:
                        if axx.get_edgecolor()[0] == 0.6509803921568628:
                            axx.remove()

    # plot spectra: group mean for word and face blocks
    for df, material, ax in zip([df_w, df_f], ['words','faces'], [ax_u2, ax_l2]):
        plot_group_spectra(df, material, ax)

    # add section titles and line between subplot rows
    ax_header.axis('off')
    for ypos in [0.48, 1.0]:
        line = plt.Line2D((0.1, 0.9), (ypos, ypos), color='black', linewidth=1, 
                        transform=fig.transFigure, figure=fig)
        fig.add_artist(line)
    fig.text(0.5, 1.03, "Word-encoding", ha='center', va='top', fontsize=7, fontdict={'fontweight': 'bold'})
    fig.text(0.5, 0.51, "Face-encoding", ha='center', va='top', fontsize=7, fontdict={'fontweight': 'bold'})

    # set titles
    for ax in [ax_u0, ax_l0]:
        ax.set_title("Task-modulated electrodes")
    for ax in [ax_u1u, ax_l1u]:
        ax.set_title("Task-modulated electrode locations")
    for ax in [ax_u2, ax_l2]:
        ax.set_title("Mean power spectra")

    # save
    plt.savefig(f"{dir_fig}/figure_3cde.png", bbox_inches='tight')
    plt.close()

    # join subplots
    join_two_figures(f"{dir_fig}/figure_3ab.png", f"{dir_fig}/figure_3cde.png",
                     f"{dir_fig}/figure_3.png", figsize=[WIDTH['2col'], 
                                                         WIDTH['2col']*3/4])


def plot_group_spectra(df, material, ax):
    if material == 'words':
        color = 'brown'
    elif material == 'faces':
        color = 'blue'

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
    
    # shade oscillation bands
    for band in ['alpha', 'gamma']:
        ax.axvspan(BANDS[band][0], BANDS[band][1], facecolor=BCOLORS[band],
                    alpha=0.4)
    
    # beautify
    ax.grid(False)
    beautify_ax(ax)

        
if __name__ == "__main__":
    main()
