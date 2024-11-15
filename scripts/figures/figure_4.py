"""
Plot results of scripts/step4_spectral_parameterization.py. For each feature of
interest, plot the distribution of values for the baseline and encoding period
as a violin (and swarm plot), with 'split' and 'dodge' set to True. Also plot
the distribution of change in each feature between the baseline and encoding 
periods as a histogram below the violin plot. 

"""

# Imports - standard
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
import seaborn as sns
from statsmodels.stats.multitest import multipletests

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from info import MATERIALS
from utils import get_start_time, print_time_elapsed
from settings import WIDTH, BANDS, PANEL_FONTSIZE
from plots import beautify_ax

# settings
plt.style.use('mplstyle/nature_neuro.mplstyle')
ALPHA = 0.05 # significance level
PLOT_SWARM = False # plot swarm plot on top of violin plot

FEATURES = ['alpha', 'gamma', 'exponent', 'alpha_adj', 'gamma_adj']
TITLES = ['Total alpha power', 'Total gamma power', 'Aperiodic exponent', 
          'Adjusted alpha power', 'Adjusted gamma power']

LABELS = ['power (\u03BCV\u00b2/Hz)', 'power (\u03BCV\u00b2/Hz)', 'exponent', 
          'power (\u03BCV\u00b2/Hz)', 'power (\u03BCV\u00b2/Hz)']

COLORS = [
    np.array([223,194,125]) / 255,
    np.array([166,97,26]) / 255,
    np.array([128,205,193]) / 255,
    np.array([1,133,113]) / 255,
]


def main():

    # display progress
    t_start = get_start_time()

    # identify / create directories
    dir_output = f"{PROJECT_PATH}/figures/main_figures"
    if not os.path.exists(dir_output): 
        os.makedirs(f"{dir_output}")

    ### load data ##############################################################
    # load SpecParam results
    fname = f"{PROJECT_PATH}/data/results/spectral_parameters.csv"
    df = pd.read_csv(fname, index_col=0)
    df = df.loc[df['memory']=='hit'].reset_index(drop=True)

    # get results for task-modulated channels
    fname = f"{PROJECT_PATH}/data/results/band_power_statistics.csv"
    temp = pd.read_csv(fname, index_col=0)
    temp = temp.loc[temp['memory']=='hit']
    df_w = df.merge(temp.loc[temp['material']=='words'], 
                    on=['patient', 'chan_idx', 'material', 'memory'])
    df_f = df.merge(temp.loc[temp['material']=='faces'], 
                    on=['patient', 'chan_idx', 'material', 'memory'])
    for df in [df_w, df_f]: # compute joint significance
        df['sig_all'] = df[[f'{band}_sig' for band in BANDS]].all(axis=1)
    df = pd.concat([df_w, df_f], ignore_index=True)
    df = df.loc[df['sig_all']].reset_index(drop=True)

    # load group-level statistical results
    fname = f"{PROJECT_PATH}/data/ieeg_stats/group_level_hierarchical_bootstrap_active.csv"
    stats = pd.read_csv(fname, index_col=0)
    stats = stats.loc[stats['memory'] =='hit'].reset_index(drop=True) # only successful memory trials
    stats['p'] = stats['pvalue'].apply(lambda x: min(x, 1-x)) # standardize p-values (currently 0.5 represents equal and <0.05 and >0.95 are both significant)
    stats.loc[stats['p']==0, 'p'] = 0.001 # set p=0 to p=0.001 (lowest possible value with 1000 bootstrap samples)
    mt = multipletests(stats['p'], alpha=0.05, method='holm') # multiple comparisons correction
    stats['p_corr'] = mt[1]
    
    ### plot ##################################################################
    # create figure and gridspec
    fig = plt.figure(figsize=[WIDTH['2col'], WIDTH['2col']], 
                     tight_layout=True)
    spec = gridspec.GridSpec(figure=fig, ncols=3, nrows=2)

    # loop features
    for i_feature, (feature, label, title) in \
        enumerate(zip(FEATURES, LABELS, TITLES)):

        # create nested gridspec for each feature
        gs = gridspec.GridSpecFromSubplotSpec(2, 2, height_ratios=[2, 0.8],
                                              subplot_spec=spec[i_feature+1])

        # plot violin
        ax_v = fig.add_subplot(gs[0, :])
        ax_v.set_title(title)
        plot_violin(ax_v, df, feature, label)

        # plot histogram
        ax_h0 = fig.add_subplot(gs[1, 0])
        ax_h1 = fig.add_subplot(gs[1, 1], sharey=ax_h0)
        ax_h0.set_ylabel('channel count')
        plot_histogram([ax_h0, ax_h1], df, feature, label)

        # annotate stats
        print_stats([ax_h0, ax_h1], stats, feature)

        # beautify axes
        for ax in [ax_v, ax_h0, ax_h1]:
            beautify_ax(ax)

    # add figure panel labels
    # fig.text(0.02, 0.98, 'a', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.34, 0.98, 'a', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.68, 0.98, 'b', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.02, 0.48, 'c', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.34, 0.48, 'd', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.68, 0.48, 'e', fontsize=PANEL_FONTSIZE, fontweight='bold')

    # save
    fig.savefig(f"{dir_output}/figure_4", bbox_inches='tight')
    fig.savefig(f"{dir_output}/figure_4.png", bbox_inches='tight')

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


def plot_violin(ax, df, feature, label):
    plotting_params = {
        'data'  :   df,
        'x'     :   'material',
        'y'     :   feature,
        'hue'   :   'epoch',
        'dodge' :   True,
        'split' :   True
    }
        
    # plot violin
    vp = sns.violinplot(**plotting_params, ax=ax)
    for l in ax.lines: # remove mean and quartile line
        l.set_linewidth(0)

    # set labels
    vp.set_ylabel(label)
    vp.set_xlabel('')
    vp.set_xticks([0, 1], ['word-encoding', 'face-encoding'], fontsize=6)

    # re-set color of violin plot
    iv = 0
    for violin in vp.collections:
        if isinstance(violin, PolyCollection):
            violin.set_facecolor(COLORS[iv])
            iv += 1

    # create legend
    legend_elements = [
        Patch(facecolor=COLORS[0], edgecolor='black', label='word, baseline'),
        Patch(facecolor=COLORS[1], edgecolor='black', label='word, encoding'),
        Patch(facecolor=COLORS[2], edgecolor='black', label='face, baseline'),
        Patch(facecolor=COLORS[3], edgecolor='black', label='face, encoding')
    ]
    ax.legend(handles=legend_elements, loc='upper center')

    # plot swarm
    if PLOT_SWARM:
        plotting_params.pop('split')
        sns.swarmplot(**plotting_params, ax=ax, size=0.5, 
                        palette='dark:#000000')


def plot_histogram(axes, df, feature, label):
    # compute difference between encoding and baseline
    df_p = df.pivot_table(index=['patient', 'chan_idx', 'material'], 
                              columns='epoch', values=feature).reset_index()
    df_p['diff'] = df_p['post'] - df_p['pre']

    # plot
    for material, ax in zip(MATERIALS, axes):
        df_m = df_p.loc[df_p['material']==material]
        max_val = np.nanmax(np.abs(df_m['diff']))
        bins = np.linspace(-max_val, max_val, 21)
        ax.hist(df_m['diff'], bins=bins, color='grey')
        ax.set(xlim=[-max_val, max_val])
        ax.set_xlabel(f"$\Delta$ {label}")
        ax.axvline(0, color='k', linestyle='-', linewidth=0.5)
        ax.axvline(np.nanmean(df_m['diff']), color='r', linewidth=0.5)


def print_stats(axes, stats, feature):
    for material, ax in zip(MATERIALS, axes):
        pval = stats.loc[((stats['material']==material) & 
                    (stats['feature']==feature)), 'p_corr'].values[0]
        sign = np.sign(stats.loc[((stats['material']==material) & 
                                  (stats['feature']==feature)), 
                                  'cohens_d'].values[0])
        if sign >= 0:
            pos = [0.58, 0.86]
        else:
            pos = [0.04, 0.86]
        if pval < 0.001:
            ax.text(*pos, f"p<0.001*", transform=ax.transAxes)
        elif pval < ALPHA:
            ax.text(*pos, f"p={pval:.3f}*", transform=ax.transAxes)
        else:
            ax.text(*pos, f"p={pval:.3f}", transform=ax.transAxes)


if __name__ == "__main__":
    main()
