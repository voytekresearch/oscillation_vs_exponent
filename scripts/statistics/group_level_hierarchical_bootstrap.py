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
    df_sig = pd.read_csv(f"{PROJECT_PATH}/data/results/ieeg_modulated_channels.csv", index_col=0)

    # init
    columns = ['material', 'memory', 'feature', 'pvalue']
    results = pd.DataFrame(columns=columns)

    # loop through materials
    for material in ['faces','words']:
        # loop through memory conditions
        for memory in ['hit','miss']:

            # display progress
            start_time_c = get_start_time()
            print('---------------------------------------')
            print(f'Analyzing condition: {material} - {memory}')
            
            # get data for condition
            df_cond = df_params.loc[(df_params['material'] == material) & \
                                    (df_params['memory']==memory)].reset_index(drop=True)
            
            # filter for task-modulated channels
            if ACTIVE_ONLY:
                df_cond = df_cond.merge(df_sig, on=['patient', 'chan_idx'])
                df_cond = df_cond.loc[df_cond['sig_all']].reset_index(drop=True)
            
            # run bootstrap
            for feature in FEATURES:
                stats = hb(df_cond, feature, 'epoch', 'patient', 'chan_idx', 
                           n_iterations=N_ITERATIONS, verbose=False, plot=False)
                data = np.array([[material, memory, feature, stats[0]]])
                results_i = pd.DataFrame(data, index=[0], columns=columns)
                results = pd.concat([results, results_i], ignore_index=True)

            # display progress
            print_time_elapsed(start_time_c)

    # save results
    if ACTIVE_ONLY:
        fname_out = "group_level_hierarchical_bootstrap_active.csv"
    else:
        fname_out = "group_level_hierarchical_bootstrap.csv"
    results.to_csv(f"{dir_output}/{fname_out}")

    # display progress
    print('\n---------------------------------------')
    print('Analysis complete!')
    print_time_elapsed(start_time)


if __name__ == "__main__":
    main()
