"""

"""

# Imports - standard
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from utils import get_start_time, print_time_elapsed
from info import PATIENTS
from settings import COLORS

# settings
plt.style.use('mplstyle/default.mplstyle')
FIGSIZE = [7.5, 4]


def main():

    # display progress
    t_start = get_start_time()

    # identify / create directories
    dir_output = f"{PROJECT_PATH}/figures/main_figures"
    if not os.path.exists(dir_output): 
        os.makedirs(f"{dir_output}")

    # load speactral results
    df = load_params()

    # run OLS
    results = {}
    df_ols_list = []
    for feature in ['alpha', 'alpha_adj', 'gamma', 'gamma_adj']:
        # OLS
        results_i = run_ols(df, feature)
        results[feature] = results_i

        # bootstrap OLS
        for patient in PATIENTS:
            results_i = run_ols(df.loc[df['patient']!=patient], feature)
            df_ols_list.append(pd.DataFrame({
                                            'patient'   :   patient, 
                                            'feature'   :   feature,
                                            'rsquared'  :   results_i.rsquared},
                                            index=[0]))
    df_ols = pd.concat(df_ols_list, axis=0).reset_index(drop=True)

    # print results
    for feature in ['alpha', 'alpha_adj', 'gamma', 'gamma_adj']:
        print(f"\n\n{feature} results:")
        print(f"\tR-squared: {results[feature].rsquared:.3f}")
        print(f"\tF-statistic: {results[feature].fvalue:.3f}")
        if results[feature].f_pvalue < .001:
            print(f"\tp-value: {results[feature].f_pvalue:.3e}")
        else:
            print(f"\tp-value: {results[feature].f_pvalue:.3f}")
        print(results[feature].summary())

    # create figure
    fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2, 3, figsize=FIGSIZE,
                                            constrained_layout=True)
    ax1.set_title('Total power v. exponent')
    ax2.set_title('Adjusted power v. exponent')
    ax3.set_title('R-squared')

    # plot scatter and regression results
    features = ['alpha', 'alpha_adj', 'gamma', 'gamma_adj']
    colors = ['brown', 'light_brown', 'blue', 'light_blue']
    for ax, feature, color in zip([ax1, ax2, ax4, ax5], features, colors):
        df.plot.scatter(y=f"{feature}_diff", x='exponent_diff', ax=ax, 
                        color=COLORS[color], s=2)
        draw_regression_results(ax, df['exponent_diff'].values, 
                                results[f'{feature}'])
        ax.axvline(0, color='grey', linestyle='--', linewidth=1)
        ax.axhline(0, color='grey', linestyle='--', linewidth=1)

    # label axes 
    ax1.set(xlabel='$\Delta$ exponent', ylabel='$\Delta$ total alpha')
    ax2.set(xlabel='$\Delta$ exponent', ylabel='$\Delta$ adjusted alpha')
    ax4.set(xlabel='$\Delta$ exponent', ylabel='$\Delta$ total gamma')
    ax5.set(xlabel='$\Delta$ exponent', ylabel='$\Delta$ adjusted gamma')

    # plot R-squared
    for feature, ax, color in zip(['alpha', 'gamma'], 
                                  [ax3, ax6], 
                                  ['brown', 'blue']):
        plotting_params = {
            'data':    df_ols.loc[((df_ols['feature']==feature) |
                                    (df_ols['feature']==f"{feature}_adj"))],
            'x':       'feature',
            'y':       'rsquared',
        }
        palette = {feature: COLORS[color], 
                   f"{feature}_adj": COLORS[f'light_{color}']}
        sns.violinplot(**plotting_params, ax=ax, palette=palette)
        sns.swarmplot(**plotting_params, color=[0,0,0], ax=ax, size=3)
        ax.set(xlabel='regressor', ylabel='R-squared')
        ax.set_xticklabels(['total power','adjusted power'])

    # save figure
    plt.savefig(f"{dir_output}/regress_confounding_features.png")

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


def draw_regression_results(ax, x_data, results):
    # regression results
    x = np.array([np.nanmin(x_data), np.nanmax(x_data)])
    y = x * results.params[1] + results.params[0]
    
    # design text based on values
    if results.f_pvalue < .001:
        str_p = f"{results.f_pvalue:.1e}"
    else:
        str_p = f"{results.f_pvalue:.3f}"
    if results.rsquared < .001:
        str_r2 = "<0.001" #"$<e^{-3}$"
    else:
        str_r2 = f"{results.rsquared:.3f}"            
    s = "$\it{R^{2}}$: " + f"{str_r2}" + "\n$\it{p}$:   " + f"{str_p}"
        
    # plot
    ax.plot(x, y, color='k')
    ax.text(0.8,0.1, s, transform=ax.transAxes, bbox=dict(facecolor='w'), fontsize=10)

    return ax


def run_ols(df, feature):
    df_i = df[['exponent_diff', f'{feature}_diff']].dropna() # drop nan
    X = sm.add_constant(df_i[[f'exponent_diff']]) # add constant term
    y = df_i[f'{feature}_diff']
    model = sm.OLS(y, X)
    results = model.fit()

    return results


def load_params():
    # load data
    df = pd.read_csv(f"{PROJECT_PATH}/data/results/spectral_parameters.csv", index_col=0)
    df = df[['patient', 'chan_idx', 'material', 'memory', 'epoch', 'exponent', 
             'alpha', 'alpha_adj', 'gamma', 'gamma_adj']] # drop unneeded columns   
    df = df.loc[df['memory']=='hit'] # drop unsuccessful trials

    # pivot table (based on epoch)
    index = ['patient', 'chan_idx', 'material']
    features = ['exponent', 'alpha', 'alpha_adj', 'gamma', 'gamma_adj']
    df = df.pivot_table(index=index, columns='epoch', values=features).reset_index()

    # compute difference in parameters
    for feature in features:
        df[f"{feature}_diff"] = df[(feature, 'post')] - df[(feature, 'pre')]

    # drop original columns
    df = df.drop(columns=[(feature, epoch) for feature in features for epoch in ['pre', 'post']])
    df.columns = df.columns.droplevel(1)

    return df


if __name__ == "__main__":
    main()
