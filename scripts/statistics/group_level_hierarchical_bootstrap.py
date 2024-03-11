"""
Group-level hierarchical bootstrap. Perform group-level comparison of 
spectral parameters using the hierarchical bootstrap.

"""

# Imports - standard
import os
import numpy as np
import pandas as pd
from timeit import default_timer as timer

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from bootstrap import run_hierarchical_bootstrap as hb

# analysis/statistical settings
N_ITERATIONS = 1000 # number of iterations for permutation test
FEATURES = ['exponent', 'alpha', 'alpha_adj', 'gamma', 'gamma_adj']


def main():
    # display progress
    start_time = timer()
    
    # id directories
    dir_output = f'{PROJECT_PATH}/data/ieeg_stats'
    if not os.path.exists(dir_output): os.makedirs(dir_output)
    
    # load data
    df_params = pd.read_csv(f"{PROJECT_PATH}/data/results/spectral_parameters.csv", index_col=0)

    # init
    columns = ['material', 'memory', 'feature', 'pvalue']
    results = pd.DataFrame(columns=columns)

    # loop through materials
    for material in ['faces','words']:
        # loop through memory conditions
        for memory in ['hit','miss']:

            # display progress
            start_time_c = timer()
            print('---------------------------------------')
            print(f'Condition: {material} - {memory}')
            print('---------------------------------------')
            
            # get data for condition
            df_cond = df_params.loc[(df_params['material'] == material) & \
                                    (df_params['memory']==memory)].reset_index(drop=True)
            
            # run bootstrap
            for feature in FEATURES:
                stats = hb(df_cond, feature, 'epoch', 'chan_idx', 'patient', 
                           n_iterations=N_ITERATIONS, verbose=False, plot=False)
                results_i = pd.DataFrame(np.array([[material, memory, feature, stats[0]]]),
                                         index=[0], columns=columns)
                results = pd.concat([results, results_i], ignore_index=True)

            # display progress
            print(f'Condition complete. Time: {timer() - start_time_c}')

    # save results
    fname_out = "group_level_hierarchical_bootstrap.csv"
    results.to_csv(f"{dir_output}/{fname_out}")

    # display progress
    print('\n---------------------------------------')
    print('Analysis complete!')
    print(f'Total time: {timer() - start_time}')
        
        

if __name__ == "__main__":
    main()
