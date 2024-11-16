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
from settings import COLORS, FREQ_RANGE, WIDTH, BCOLORS, BANDS, PANEL_FONTSIZE
from info import MATERIALS

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

    # create nested gridspec (2 rows that will be further segmented)
    fig = plt.figure(figsize=(WIDTH['2col'], WIDTH['2col']*1.2))
    gs = gridspec.GridSpec(figure=fig, ncols=1, nrows=3, 
                           height_ratios=[1, 0.1, 1])
    for i_mat, [material, i_gs], in enumerate(zip(MATERIALS, [0, 2])):
        spec = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[i_gs],
                                                width_ratios=[1, 3])

        # plot barchart: number of task-modulated electrodes ###################
        ax_bar = fig.add_subplot(spec[0, 0])
        plot_barchart(df, ax_bar)
        
        # plot glass brain: electrode locations ################################
        for i_band, band in enumerate(BANDS.keys()):
            ax_brain = fig.add_subplot(spec[i_band, 1])
            boxb = ax_brain.get_position()
            expand = 1.4
            y_shift = 0.07
            x_shift = -0.05
            if (i_band==0) and (i_mat==0):
                ax_brain.set_position([boxb.x0+x_shift, boxb.y0+y_shift, 
                                       boxb.width*expand, boxb.height*expand])
            elif (i_band==1) and (i_mat==1):
                ax_brain.set_position([boxb.x0+x_shift, boxb.y0-y_shift, 
                                       boxb.width*expand, boxb.height*expand])
            else:
                ax_brain.set_position([boxb.x0+x_shift, boxb.y0, 
                                       boxb.width*expand, boxb.height*expand])
            plot_glass_brain(ax_brain, df, band)

        # plot spectra: group mean for word and face blocks #
        ax_psd = fig.add_subplot(spec[1, 0])
        # boxb = ax_psd.get_position()
        # ax_psd.set_position([boxb.x0, boxb.y0, boxb.width, boxb.height*0.9])
        plot_group_spectra(df, material, ax_psd)

        # # beautify axes
        # for ax in [ax_bar, ax_psd]:
        for ax in [ax_bar]:
            beautify_ax(ax)

    # add section titles and line between subplot rows
    for ypos in [0.5, 1.02]:
        line = plt.Line2D((0.1, 0.9), (ypos, ypos), color='black', linewidth=1, 
                        transform=fig.transFigure, figure=fig)
        fig.add_artist(line)
    fig.text(0.5, 1.04, "Word-encoding", ha='center', va='top', fontsize=7, 
             fontdict={'fontweight': 'bold'})
    fig.text(0.5, 0.52, "Face-encoding", ha='center', va='top', fontsize=7, 
             fontdict={'fontweight': 'bold'})

    # # add figure panel labels
    fig.text(0.01, 1.00, 'c', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.01, 0.75, 'd', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.32, 1.00, 'e', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.01, 0.48, 'f', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.01, 0.23, 'g', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.32, 0.48, 'h', fontsize=PANEL_FONTSIZE, fontweight='bold')

    # save
    plt.savefig(f"{dir_fig}/figure_3c-h.png", bbox_inches='tight')
    plt.close()

    # join subplots
    join_two_figures(f"{dir_fig}/figure_3ab.png", f"{dir_fig}/figure_3c-h.png",
                     f"{dir_fig}/figure_3.png", figsize=[WIDTH['2col'], 
                                                         WIDTH['2col']*1.25])


def plot_barchart(df, ax):
    # plot
    x = [0, 1, 2]
    y = [df[col].sum() / len(df) * 100 for col in ['alpha_sig', 'gamma_sig', 
                                                   'sig_all']] 
    ax.bar(x, y, color=[BCOLORS['alpha'], BCOLORS['gamma'], 'k'],
            edgecolor='black', linewidth=1, width=1)
    
    # label
    ax.set_title("Task-modulated electrodes")
    ax.set_xticks(x, labels=['alpha', 'gamma', 'both'])
    ax.set_ylabel('percentage (%)')
    ax.set_xlabel('frequency band')


def plot_glass_brain(ax, df, band, node_size=1.5):

    # shift subplot spaceing (nilearn plot including inexplicable whitespace)
    coords = df.loc[df[f'{band}_sig'], ['pos_x', 'pos_y', 'pos_z']].values
    nfig = plotting.plot_markers(axes=ax, node_coords=coords, 
                                node_values=np.ones(len(coords)), 
                                node_cmap=ListedColormap([BCOLORS[band]]),
                                display_mode='ortho', colorbar=False, 
                                annotate=False, node_size=node_size, 
                                node_kwargs={'alpha' : 1}, alpha=1)
    coords = df.loc[df[f'sig_all'], ['pos_x', 'pos_y', 'pos_z']].values
    nfig.add_markers(marker_coords=coords, marker_size=node_size, 
                        marker_color='k', alpha=1)
    nfig.annotate(size=7) # must plot with annotate=False, then set size here

    # remove gyri/sulci lines
    for xyz in nfig.axes:
        for axx in nfig.axes[xyz].ax.get_children():
            if type(axx) == mpl.patches.PathPatch:
                if axx.get_edgecolor()[0] == 0.6509803921568628:
                    axx.remove()


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

        
if __name__ == "__main__":
    main()
