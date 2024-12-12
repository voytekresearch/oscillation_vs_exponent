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
from settings import *
from info import MATERIALS

# settings
plt.style.use(MPLSTYLE)
NODE_SIZE = 2.5


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
                                                width_ratios=[1, 3], 
                                                height_ratios=[1, 1.5])

        # plot barchart: number of task-modulated electrodes ###################
        ax_bar = fig.add_subplot(spec[0, 0])
        plot_barchart(df, ax_bar)
        
        # plot glass brain: electrode locations ################################

        # create axes and adjust position for inexplicable whitespace issue with Nilearn
        ax_0 = fig.add_subplot(spec[0, 1])
        boxb = ax_0.get_position()
        expand = 1.4
        y_shift = 0.07
        x_shift = -0.05
        if i_mat==0:
            ax_0.set_position([boxb.x0+x_shift, boxb.y0+y_shift, 
                                    boxb.width*expand, boxb.height*expand])
        else:
            ax_0.set_position([boxb.x0+x_shift, boxb.y0, 
                                    boxb.width*expand, boxb.height*expand])

        plot_glass_brain(ax_0, df, node_size=NODE_SIZE)

        # plot spectra #########################################################
        # create nested gridspec for spectra (1x3)
        spec1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=spec[1, :], 
                                                 width_ratios=[1, 1, 1])
        
        # plot spectra - group mean
        ax_psd0 = fig.add_subplot(spec1[0])
        plot_group_spectra(df, material, ax_psd0)
        ax_psd0.set_title("Average power spectra")

        # plot spectra - difference in group mean
        ax_psd1 = fig.add_subplot(spec1[1])
        plot_group_spectra_diff(df, material, ax_psd1)

        # plot barchart showing total power differnces in each band
        ax_psd2 = fig.add_subplot(spec1[2])
        ax_psd2.set_title("Difference in total power")
        for band in ['alpha', 'gamma']:
            y = np.log10(df[f'{band}_post']) - np.log10(df[f'{band}_pre'])
            ax_psd2.bar(band, y.mean(), color=BCOLORS[band], edgecolor='black', 
                        linewidth=1)
        ax_psd2.set_ylabel('log power (\u03BCV\u00b2/Hz)')
        ax_psd2.set_xlabel('frequency band')
        ax_psd2.axhline(0, color='grey', linestyle='--', linewidth=1)

        # # beautify axes
        for ax in [ax_bar, ax_psd0, ax_psd1]:
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
    fig.text(0.02, 1.00, 'c', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.32, 1.00, 'd', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.02, 0.76, 'e', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.34, 0.76, 'f', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.68, 0.76, 'g', fontsize=PANEL_FONTSIZE, fontweight='bold')

    fig.text(0.02, 0.48, 'h', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.32, 0.48, 'i', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.02, 0.23, 'j', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.34, 0.23, 'k', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.68, 0.23, 'l', fontsize=PANEL_FONTSIZE, fontweight='bold')

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
            edgecolor='black', linewidth=1)
    
    # label
    ax.set_title("Task-modulated electrodes")
    ax.set_xticks(x, labels=['alpha', 'gamma', 'both'])
    ax.set_ylabel('percentage (%)')
    ax.set_xlabel('frequency band')


def plot_glass_brain(ax, df, node_size=1.5):
    coords = df.loc[df[f'alpha_sig'], ['pos_x', 'pos_y', 'pos_z']].values
    nfig = plotting.plot_markers(axes=ax, node_coords=coords, 
                                node_values=np.ones(len(coords)), 
                                node_cmap=ListedColormap([BCOLORS['alpha']]),
                                display_mode='ortho', colorbar=False, 
                                annotate=False, node_size=node_size, alpha=1)

    coords = df.loc[df[f'gamma_sig'], ['pos_x', 'pos_y', 'pos_z']].values
    nfig.add_markers(marker_coords=coords, marker_size=node_size, 
                        marker_color=BCOLORS['gamma'], alpha=1)

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


def plot_group_spectra_diff(df, material, ax):

    # load data
    fname = f"{PROJECT_PATH}/data/ieeg_spectral_results/psd_{material}_hit_XXXstim.npz"
    data_pre = np.load(fname.replace("XXX", "pre"))
    data_post = np.load(fname.replace("XXX", "post"))
    psd_pre = data_pre['spectra'][df[f"sig_all"]]
    psd_post = data_post['spectra'][df[f"sig_all"]]
    freq = data_pre['freq']

    # plot
    f_mask = np.logical_and(freq>FREQ_RANGE[0], freq<FREQ_RANGE[1])
    psd_diff = np.log10(psd_post[:, f_mask]) - np.log10(psd_pre[:, f_mask])
    plot_psd_diff(freq[f_mask], psd_diff, ax)
    
    # shade oscillation bands
    for band in ['alpha', 'gamma']:
        ax.axvspan(BANDS[band][0], BANDS[band][1], facecolor=BCOLORS[band],
                    alpha=0.4)
    
    # beautify
    ax.grid(False)


def plot_psd_diff(freq, psd_diff, ax):
    """ 
    Plot spectra (or change in spectral power) in semi-log space.
    The mean spectrum is plotted in black, and the individual spectra are plotted in grey.
    A horizontal line at power=0 is also plotted.

    Parameters
    ----------
    freq : array
        Frequency values.
    psd_diff : array
        Spectral power values (difference in log power between 2 spectra).
    ax : matplotlib axis, optional
        Axis to plot on. If None, a new figure is created.

    Returns
    -------
    None.
    
    """

    # plot mean
    ax.plot(freq, np.nanmean(psd_diff, axis=0), color='k', linewidth=3,
                label="mean")

    # shade sem
    mean = np.nanmean(psd_diff, axis=0)
    sem = np.nanstd(psd_diff, axis=0) / np.sqrt(psd_diff.shape[0])
    ax.fill_between(freq, mean - sem, mean + sem,
                    color='k', alpha=0.2, label="SEM")

    # scale x-axis logarithmically
    ax.set(xscale="log")

    # set axes ticks and labels
    ax.set_title(f"Difference in power (encoding - baseline)")
    ax.set_ylabel('log power (\u03BCV\u00b2/Hz)')
    ax.set_xlabel('frequency (Hz)')
    ax.set_xticks([10, 100])
    ax.set_xticklabels(["10", "100"])

    # annotate power=0
    ax.axhline(0, color='grey', linestyle='--', linewidth=1)

        
if __name__ == "__main__":
    main()
