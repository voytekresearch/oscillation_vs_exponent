"""
Perform group-level comparison of spectral parameters between baseline and 
encoding using the paired hierarchical bootstrap.

"""

# Imports - standard
import os
import numpy as np
import pandas as pd

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from paired_hierarchical_bootstrap import hierarchical_bootstrap as hb
from utils import get_start_time, print_time_elapsed
from settings import BANDS

# analysis/statistical settings
N_ITERATIONS = 1000 # number of iterations for permutation test
FEATURES = ['exponent', 'alpha', 'alpha_adj', 'gamma', 'gamma_adj']
ACTIVE_ONLY = True # whether to analyze task-modulated channels only

def main():
    # display progress
    start_time = get_start_time()

    # id directories
    dir_output = f'{PROJECT_PATH}/data/ieeg_stats'
    if not os.path.exists(dir_output): os.makedirs(dir_output)

    # load data
    df_params = pd.read_csv(f"{PROJECT_PATH}/data/results/spectral_parameters.csv", index_col=0)
    df_sig = pd.read_csv(f"{PROJECT_PATH}/data/results/band_power_statistics.csv", index_col=0)

    # init
    columns = ['material', 'feature', 'pvalue']
    results = pd.DataFrame(columns=columns)

    # loop through materials
    for material in ['words', 'faces']:

        # display progress
        start_time_c = get_start_time()
        print('---------------------------------------')
        print(f'Analyzing {material}-encoding')
        
        # get data for condition
        df_cond = df_params.loc[df_params['material'] == material].reset_index(drop=True).drop(columns=['material'])
        
        # filter for task-modulated channels
        if ACTIVE_ONLY:
            temp = df_sig.loc[(df_sig['material']==material) & \
                              (df_sig['memory']=='hit')].reset_index(drop=True).drop(columns=['material','memory'])
            temp['sig_all'] = temp[[f'{band}_sig' for band in BANDS]].all(axis=1)
            df_cond = df_cond.merge(temp, on=['patient', 'chan_idx'])
            df_cond = df_cond.loc[df_cond['sig_all']].reset_index(drop=True)
        
        # run bootstrap for each feature
        for feature in FEATURES:
            # display progress
            print(f'\nFeature: {feature}')
            start_time_f = get_start_time()

            # compute difference between baseline and encoding
            df_f = df_cond.pivot_table(index=['patient','chan_idx','memory'], 
                                        columns='epoch', values=feature).reset_index()
            df_f[feature] = df_f['post'] - df_f['pre']

            # run bootstrap
            stats = hb(df_f, feature, 'memory', 'patient', 'chan_idx',
                        n_iterations=N_ITERATIONS, verbose=True, plot=False)
            data = np.array([[material, feature, stats[0]]])
            results_i = pd.DataFrame(data, index=[0], columns=columns)
            results = pd.concat([results, results_i], ignore_index=True)

            # display progress
            print(f"p-value: {stats[0]}")
            print_time_elapsed(start_time_f)  

        # display progress
        print_time_elapsed(start_time_c)

    # save results
    if ACTIVE_ONLY:
        fname_out = "group_level_sme_active.csv"
    else:
        fname_out = "group_level_sme.csv"
    results.to_csv(f"{dir_output}/{fname_out}")

    # display progress
    print('\n---------------------------------------')
    print('Analysis complete!')
    print_time_elapsed(start_time)


if __name__ == "__main__":
    main()
