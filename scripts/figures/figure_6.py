"""

"""

# Imports - standard
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from utils import get_start_time, print_time_elapsed
from info import PATIENTS
from settings import BCOLORS, WIDTH
from plots import beautify_ax

# settings
plt.style.use('mplstyle/default.mplstyle')


def main():

    # display progress
    t_start = get_start_time()

    # identify / create directories
    dir_figure = f"{PROJECT_PATH}/figures/main_figures"
    dir_output = f"{PROJECT_PATH}/data/ieeg_stats"
    for path in [dir_figure, dir_output]:
        if not os.path.exists(path): 
            os.makedirs(path)

    # load speactral results
    df = load_params()
    df = df.loc[df['material']=='faces']

    # run OLS
    results = {}
    df_ols_list = []
    for feature in ['alpha', 'alpha_adj', 'gamma', 'gamma_adj']:
        # OLS
        results_i = run_ols(df, feature)
        results[feature] = results_i

        # cross-validate OLS
        for patient in PATIENTS:
            results_i = run_ols(df.loc[df['patient']==patient], feature)
            df_ols_list.append(pd.DataFrame({
                                            'patient'   :   patient, 
                                            'feature'   :   feature,
                                            'rsquared'  :   results_i.rsquared},
                                            index=[0]))
    df_ols = pd.concat(df_ols_list, axis=0).reset_index(drop=True)
    df_ols.to_csv(f"{dir_output}/confounding_features_ols.csv")

    # run t-test on cross-validation R-squared values and print results
    sig = {}
    for feature in ['alpha', 'gamma']:
        r2_total = df_ols.loc[df_ols['feature']==feature, 'rsquared'].values
        r2_adj = df_ols.loc[df_ols['feature']==f"{feature}_adj", 'rsquared'].values
        t, p = ttest_rel(r2_total, r2_adj)
        sig[feature] = p < .05
        print(f"\n{feature} cross-validation t-test results:")
        print(f"\tT-statistic: {t:.3f}")
        if p < .001:
            print(f"\tp-value: {p:.1e}")
        else:
            print(f"\tp-value: {p:.3f}")

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
    figsize = [WIDTH['2col'], WIDTH['2col']/2]
    _, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2, 3, figsize=figsize,
                                                    width_ratios=[1,1,1],
                                                    constrained_layout=True)
    ax2.sharey(ax1)
    ax5.sharey(ax4)
    ax4.sharex(ax1)
    ax5.sharex(ax2)
    
    # set titles
    ax1.set_title('Total power')
    ax2.set_title('Adjusted power')
    ax3.set_title('Linear model: power v. exponent')

    # plot scatter and regression results
    features = ['alpha', 'alpha_adj', 'gamma', 'gamma_adj']
    for ax, feature in zip([ax1, ax2, ax4, ax5], features):
        df.plot.scatter(y=f"{feature}_diff", x='exponent_diff', ax=ax, 
                        color=BCOLORS[feature.split('_')[0]], s=2, alpha=0.5)
        draw_regression_results(ax, df['exponent_diff'].values, 
                                results[f'{feature}'], add_text=False)
        ax.axvline(0, color='grey', linestyle='--', linewidth=1)
        ax.axhline(0, color='grey', linestyle='--', linewidth=1)

    # label axes 
    ax1.set(ylabel='$\Delta$ alpha')
    ax4.set(xlabel='$\Delta$ exponent', ylabel='$\Delta$ gamma')
    ax5.set(xlabel='$\Delta$ exponent')
    
    # plot R-squared
    for feature, ax in zip(['alpha', 'gamma'], 
                                       [ax3, ax6]):
        plotting_params = {
            'data':    df_ols.loc[((df_ols['feature']==feature) |
                                    (df_ols['feature']==f"{feature}_adj"))],
            'x':       'feature',
            'y':       'rsquared',
        }
        sns.boxplot(**plotting_params, ax=ax, color=BCOLORS[feature])
        sns.swarmplot(**plotting_params, color=[0,0,0], ax=ax, size=3)
        ax.set_ylabel('$R^{2}$')
        ax.set_xticks([0, 1], labels=['total\npower','adjusted\npower'])
    ax3.set_xlabel('')
    ax6.set_xlabel('regressor')

    # beautify axes
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        beautify_ax(ax)

    # save figure
    fname = "figure_6"
    plt.savefig(f"{dir_figure}/{fname}.png")
    plt.savefig(f"{dir_figure}/{fname}")

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


def draw_regression_results(ax, x_data, results, add_text=True):
    # regression results
    x = np.array([np.nanmin(x_data), np.nanmax(x_data)])
    y = x * results.params[1] + results.params[0]
    
    # plot
    ax.plot(x, y, color='k')

    # add text
    if add_text:
        if results.f_pvalue < .001:
            str_p = f"{results.f_pvalue:.1e}"
        else:
            str_p = f"{results.f_pvalue:.3f}"
        if results.rsquared < .001:
            str_r2 = "<0.001" #"$<e^{-3}$"
        else:
            str_r2 = f"{results.rsquared:.3f}"            
        s = "$\it{R^{2}}$: " + f"{str_r2}" + "\n$\it{p}$:   " + f"{str_p}"
        ax.text(0.02, 0.78, f"r = {str_r2}\np = {str_p}",
                transform=ax.transAxes)

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
    df = df.drop(columns=[(feature, epoch) for feature in features for epoch in 
                          ['pre', 'post']])
    df.columns = df.columns.droplevel(1)

    return df


if __name__ == "__main__":
    main()
