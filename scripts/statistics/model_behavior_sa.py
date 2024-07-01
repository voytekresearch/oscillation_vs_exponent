"""
Sensitive analysis for scripts/statistics/model_behavior.py

"""

# Imports - standard
import os
import numpy as np
import pandas as pd

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from info import PATIENTS
from utils import get_start_time, print_time_elapsed
from model_behavior import load_params, run_logistic_regression_cv


def main():

    # display progress
    t_start = get_start_time()

    # identify / create directories
    dir_output = f"{PROJECT_PATH}/data/specparam_sensitivity_analysis/logit_scores"
    if not os.path.exists(dir_output): 
        os.makedirs(f"{dir_output}")

    # loop through results for each parameter set
    dir_input = f"{PROJECT_PATH}/data/specparam_sensitivity_analysis"
    files = [f for f in os.listdir(dir_input) if f.endswith('.csv')]
    for i_file, fname in enumerate(files):
        # display progress
        print(f"\n\nAnalyzing {fname} ({i_file+1}/{len(files)})...")
        t_start_i = get_start_time()

        # load results
        results = load_params(f"{dir_input}/{fname}")
        results['memory'] = results['memory'].map({'hit': 1, 'miss': 0})

        # run logistic regression for all channels in each trial condition
        features_x = ['exponent_diff', 'alpha_diff', 'gamma_diff']

        # create dataframe to store results (one row for each channel-material pair)
        df = results.groupby(['patient','channel','material','ap_mode']).sum().reset_index()
        df = df[['patient', 'channel', 'material', 'ap_mode']]
        df['score'] = np.nan 

        # run logistic regression 
        for patient in PATIENTS:
            channels = results.loc[results['patient']==patient, 'channel'].unique()
            for channel in channels:
                for material in ['words', 'faces']:
                    for ap_mode in ['fixed', 'knee']:
                        try: # some patients/channels don't have data
                            res_i = results.loc[(results['patient']==patient) & \
                                                (results['channel']==channel) & \
                                                (results['material']==material) & \
                                                (results['ap_mode']==ap_mode)]
                            _, score = run_logistic_regression_cv(res_i, features_x, 
                                                                'memory')
                            df_index = (df['patient']==patient) & \
                                        (df['channel']==channel) & \
                                        (df['material']==material) & \
                                        (df['ap_mode']==ap_mode)
                            df.loc[df_index, 'score'] = score
                        except:
                            pass

        # save results
        df.to_csv(f"{dir_output}/{fname}")

        # display progress
        print(f"Analysis time for {fname}:")
        print_time_elapsed(t_start_i)

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


if __name__ == "__main__":
    main()
