"""
Plot results of scripts/step4_spectral_parameterization.py. For each feature of
interest, plot the distribution of values for the baseline and encoding period
as a violin and swarm plot, with 'split' and 'dodge' set to True. Also plot the
distribution of change in each feature between the baseline and encoding periods
as a histogram below the violin plot. 

"""

# Imports - standard
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from statsmodels.stats.multitest import multipletests

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from info import MATERIALS
from utils import get_start_time, print_time_elapsed
from plots import set_custom_colors
from settings import WIDTH, BANDS, PANEL_FONTSIZE

# settings
plt.style.use('mplstyle/nature_neuro.mplstyle')
ALPHA = 0.05 # significance level
PLOT_SWARM = False # plot swarm plot on top of violin plot

FEATURES = ['exponent', 'alpha_adj', 'gamma_adj']
TITLES = ['Aperiodic exponent', 'Adjusted alpha power', 'Adjusted gamma power']
LABELS = ['exponent', 'power (\u03BCV\u00b2/Hz)', 'power (\u03BCV\u00b2/Hz)']


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

    # load group-level statistical results
    fname = f"{PROJECT_PATH}/data/ieeg_stats/group_level_hierarchical_bootstrap_active.csv"
    stats = pd.read_csv(fname, index_col=0)
    stats = stats.loc[stats['memory'] =='hit'].reset_index(drop=True)
    stats['p'] = stats['pvalue'].apply(lambda x: min(x, 1-x)) # standardize p-values (currently 0.5 represents equal and <0.05 and >0.95 are both significant)
    stats.loc[stats['p']==0, 'p'] = 0.001 # set p=0 to p=0.001
    mt = multipletests(stats['p'], alpha=0.05, method='holm')
    stats['p_corr'] = mt[1]
    
    ### plot ##################################################################

    # create figure and gridspec
    fig = plt.figure(figsize=[WIDTH['2col'], WIDTH['2col']/2], 
                     constrained_layout=True)
    spec = gridspec.GridSpec(figure=fig, ncols=6, nrows=2, 
                             height_ratios=[2.5,1])

    # loop features
    x_positions = [0.17, 0.5, 0.85]
    for i_feature, (feature, label, title, xpos) in \
        enumerate(zip(FEATURES, LABELS, TITLES, x_positions)):

        # add feature label above pairs of subplots
        fig.text(xpos, 0.975, title, ha='center', va='center', fontsize=7)

        # loop materials
        for i_material, (material, df, plot_color) in \
            enumerate(zip(MATERIALS, [df_w, df_f], ['browns', 'blues'])):

            # get sig chans only
            df = df.loc[df['sig_all']].reset_index(drop=True)

            # create subplots
            ax_v = fig.add_subplot(spec[0, i_feature*2+i_material])
            ax_h = fig.add_subplot(spec[1, i_feature*2+i_material])
            
            # set plotting params
            set_custom_colors(plot_color) # set color palette
            plotting_params = {
                'data'  :   df,
                'hue'   :   'epoch',
                'y'     :   feature,
                'dodge' :   True,
                'split' :   True
            }

            # plot violin
            vp = sns.violinplot(**plotting_params, ax=ax_v)
            for l in ax_v.lines: # remove mean and quartile line
                l.set_linewidth(0)
            vp.set_xlabel('')
            vp.set_ylabel(label)
            vp.set_title(f"\n{material[:-1]}-encoding")
            vp.set_xticks([])
            ax_v.legend(loc='upper left', bbox_to_anchor=(0, 0))

            # plot swarm
            if PLOT_SWARM:
                plotting_params.pop('split')
                sns.swarmplot(**plotting_params, ax=ax_v, size=0.5, 
                                palette='dark:#000000')
                
            # plot histogram
            plot_histogram(ax_h, ax_v, df, feature, label, stats, material)

    # add figure panel labels
    fig.text(0.02, 0.98, 'a.', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.34, 0.98, 'b.', fontsize=PANEL_FONTSIZE, fontweight='bold')
    fig.text(0.68, 0.98, 'c.', fontsize=PANEL_FONTSIZE, fontweight='bold')

    # save
    fig.savefig(f"{dir_output}/figure_4", bbox_inches='tight')
    fig.savefig(f"{dir_output}/figure_4.png", bbox_inches='tight')

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


def plot_histogram(ax, ax_v, df, feature, label, stats, material):
    # compute difference
    df_p = df.pivot_table(index=['patient', 'chan_idx', 'material'], 
                              columns='epoch', values=feature).reset_index()
    df_p['diff'] = df_p['post'] - df_p['pre']

    # plot
    max_val = np.nanmax(np.abs(df_p['diff']))
    bins = np.linspace(-max_val, max_val, 21)
    ax.hist(df_p['diff'], bins=bins, color='grey')
    ax.set(xlim=[-max_val, max_val])
    ax.set_xlabel(f"$\Delta$ {label}")
    # ax.set_ylabel('channel count')
    ax.axvline(0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(np.nanmean(df_p['diff']), color='r', linewidth=0.5)

    # add stats
    pval = stats.loc[((stats['material']==material) & 
                (stats['feature']==feature)), 'p_corr'].values
    pos = [0.6, -0.1]
    if pval < 0.001:
        ax_v.text(*pos, f"$*p<0.001", transform=ax_v.transAxes)
    elif pval < ALPHA:
        ax_v.text(*pos, f"*p={pval[0]:.3f}", transform=ax_v.transAxes)
    else:
        ax_v.text(*pos, f"p={pval[0]:.3f}", transform=ax_v.transAxes)

if __name__ == "__main__":
    main()
