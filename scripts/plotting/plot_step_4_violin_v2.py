"""
Plot results of scripts/step4_spectral_parameterization.py. For each feature of
interest, plot the distribution of values for the baseline and encoding period
as a violin and swarm plot, with 'split' and 'dodge' set to True. Also plot the
distribution of change in each feature between the baseline and encoding periods
as a histogram below the violin plot. Here, we plot each material (words and
faces seperately). We plot the exponent, adjusted alpha power, and adjusted
gamma power.

"""

# Imports - standard
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from info import MATERIALS
from utils import get_start_time, print_time_elapsed
from plots import set_custom_colors
from specparam_utils import knee_freq

# settings
plt.style.use('mplstyle/default.mplstyle')
FIGSIZE = [2, 4]
ALPHA = 0.05 # significance level
FEATURES = ['exponent', 'alpha_adj', 'gamma_adj']
TITLES = [ 'Aperiodic exponent', 'Adjusted alpha', 'Adjusted gamma']
LABELS = ['exponent', 'power (a.u.)', 'power (a.u.)']


def main():

    # display progress
    t_start = get_start_time()

    # identify / create directories
    dir_output = f"{PROJECT_PATH}/figures/param_violin"
    if not os.path.exists(dir_output): 
        os.makedirs(f"{dir_output}")

    # load SpecParam results
    fname = f"{PROJECT_PATH}/data/results/spectral_parameters.csv"
    df = pd.read_csv(fname, index_col=0)
    df = df.loc[df['memory']=='hit'].reset_index(drop=True)
    df['knee'] = knee_freq(df['knee'], df['exponent']) # convert knee to Hz

    # get results for task-modulated channels
    fname = f"{PROJECT_PATH}/data/results/ieeg_modulated_channels.csv"
    temp = pd.read_csv(fname, index_col=0)
    df = df.merge(temp, on=['patient', 'chan_idx'])
    df = df.loc[df['sig_all']].reset_index(drop=True)

    # load group-level statistical results
    fname = f"{PROJECT_PATH}/data/ieeg_stats/group_level_hierarchical_bootstrap_active.csv"
    stats = pd.read_csv(fname, index_col=0)
    stats = stats.loc[stats['memory'] =='hit'].reset_index(drop=True)
    stats['p'] = stats['pvalue'].apply(lambda x: min(x, 1-x)) # standardize p-values (currently 0.5 represents equal and <0.05 and >0.95 are both significant)
    
    # plot
    for material, color in zip(MATERIALS, ['browns', 'blues']):
        for feature, title, label in zip(FEATURES, TITLES, LABELS):
            fname_out = f"param_violin_{material}_{feature}.png"
            set_custom_colors(color)
            plot_contrasts_violin(df.loc[df['material']==material], 
                                  stats.loc[stats['material']==material],
                                  feature, title=title, y_label=label,
                                  fname_out=f"{dir_output}/{fname_out}")

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)

        
def plot_contrasts_violin(params, stats, y_var, title='', y_label=None, 
                          fname_out=None, plot_swarm=True, loc='left'):
    # set plotting params
    plotting_params = {
        'data'  :   params,
        'hue'   :   'epoch',
        'y'     :   y_var,
        'dodge' :   True,
        'split' :   True
    }

    # init
    if y_label is None:
        y_label = y_var.lower()

    # create figure
    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(2,1, height_ratios=[2,1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # ===== Upper Subplot =====
    # plot violin and swarm
    vp = sns.violinplot(**plotting_params, ax=ax1)
    if plot_swarm:
        plotting_params.pop('split')
        sns.swarmplot(**plotting_params, ax=ax1, size=2, 
                        palette='dark:#000000')

    # remove mean and quartile line
    for l in ax1.lines:
        l.set_linewidth(0)

    # Label
    ax1.set_title(title)
    vp.set_xlabel('')
    vp.set_ylabel(y_label)
    vp.xaxis.set_ticks_position('top')

    # add legend (and add space)
    handles, _ = vp.get_legend_handles_labels()
    vp.legend(handles=handles, labels=['baseline','encoding'])
    ax1.set_xlim([-0.5, 0.6])
        
    # connect paired data points on swarm plot
    params_p = params.pivot_table(index=['patient', 'chan_idx'],
                                    columns='epoch', values=y_var).reset_index()
    data = params_p[['pre', 'post']]
    for i_chan in range(data.shape[0]):
        ax1.plot([-0.2, 0.2], data.iloc[i_chan], color='k', 
                    alpha=0.5, lw=0.5)

    # ===== Lower Subplot =====
    # plot disributions exponent change)
    df_p = params.pivot_table(index=['patient', 'chan_idx'], 
                              columns='epoch', values=y_var).reset_index()
    df_p['diff'] = df_p['post'] - df_p['pre']
    max_val = np.nanmax(np.abs(df_p['diff']))
    bins = np.linspace(-max_val, max_val, 20)
    ax2.hist(df_p['diff'], bins=bins, color='grey')
    ax2.set(xlim=[-max_val, max_val])
    ax2.set_xlabel(f"$\Delta$ {y_label}")
    ax2.set_ylabel('channel count')
    ax2.axvline(0, color='k')
    ax2.axvline(np.nanmean(df_p['diff'] ), color='r', linestyle='--')

    # add stats
    if np.nanmean(df_p['diff']) >= 0:
        p_pos = 0.05
    else:
        p_pos = 0.65
    pval = stats.loc[stats['feature']==y_var, 'p'].values
    if len(pval) == 1:
        if pval < 0.001:
            ax2.text(p_pos, 0.85, f"*p<0.001", transform=ax2.transAxes)
        elif pval < ALPHA:
            ax2.text(p_pos, 0.85, f"*p={pval[0]:.3f}", transform=ax2.transAxes)
        else:
            ax2.text(p_pos, 0.85, f"p={pval[0]:.3f}", transform=ax2.transAxes)
    else:
        print(f"Warning: missing or multiple p-values for '{y_var}'")

    # adjust axis labels
    # ax2l.set_ylim([0, ax2l.get_ylim()[1]+1])
    ax2.set_ylabel('channel count')
    for ax in [ax1, ax2]:
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

    # save figure
    if fname_out: 
        fig.savefig(fname_out)
        plt.close('all')
        
    return vp


if __name__ == "__main__":
    main()
