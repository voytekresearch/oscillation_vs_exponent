"""
This script runs a logistic regression model to predict behavior from 
spectral parameters.
"""

# Imports - standard
import os
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from info import PATIENTS
from utils import get_start_time, print_time_elapsed


def main():

    # display progress
    t_start = get_start_time()

    # identify / create directories
    dir_output = f"{PROJECT_PATH}/data/results"
    if not os.path.exists(dir_output): 
        os.makedirs(f"{dir_output}")

    # load behavioral data and single-trial specparam results
    metadata = pd.read_csv(f"{PROJECT_PATH}/data/ieeg_metadata/metadata.csv")
    results = pd.read_csv(f"{PROJECT_PATH}/data/results/psd_trial_params.csv", index_col=0)
    results['material'] = results['material'] + 's'
    results = results.merge(metadata, on=['patient', 'material', 'trial'])

    # run logistic regression for all channels in each trial condition
    features_x = ['exp_diff', 'alpha_diff', 'gamma_diff']

    # create dataframe to store results (one row for each channel-material pair)
    df = results.groupby(['patient','channel','material']).mean().reset_index()
    df = df[['patient', 'channel', 'material']]
    df['score'] = np.nan 

    # run logistic regression 
    for patient in PATIENTS:
        channels = results.loc[results['patient']==patient, 'channel'].unique()
        for channel in channels:
            for material in ['words', 'faces']:
                try: # some patients/channels don't have data
                    res_i = results.loc[(results['patient']==patient) & \
                                        (results['channel']==channel) & \
                                        (results['material']==material)]
                    _, score = run_logistic_regression_cv(res_i, features_x, 
                                                          'behavior')
                    df_index = (df['patient']==patient) & \
                                (df['channel']==channel) & \
                                (df['material']==material)
                    df.loc[df_index, 'score'] = score
                except:
                    pass

    # save results
    df.to_csv(f"{dir_output}/logistic_regression_scores_cv_.csv")

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


def run_logistic_regression_cv(results, features_x, feature_y):
    """
    Run logistic regression to predict behavior from spectral parameters,
    for a given subject/channel/material, using cross-validation.

    Parameters
    ----------
    results : pandas DataFrame
        Dataframe containing spectral parameters and behavior for each subject.
    features_x : list of str
        Feature names to use for prediction.
    feature_y : str
        Target feature to predict.  

    Returns
    -------
    clf : sklearn LogisticRegression object
        Logistic regression classifier.
    score : float
        Accuracy score for the classifier.
    """

    # get data
    X = results[features_x].to_numpy()
    y = results[feature_y].to_numpy()

    # remove nans
    nan_idx = np.isnan(X).any(axis=1) | np.isnan(y)
    X = X[~nan_idx]
    y = y[~nan_idx]

    # z-score each feature
    X = zscore(X, axis=0)

    # run logistic regression with cross-validation
    clf = LogisticRegression(random_state=0)
    scores = cross_val_score(clf, X, y, cv=5)
    score = scores.mean()

    return clf, score


if __name__ == "__main__":
    main()
