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
from settings import AP_MODE
from utils import get_start_time, print_time_elapsed


# settings
class_weights = ["balanced",
                {0:0.5, 1:0.5}, #expected 1:1
                {0:0.67, 1:0.33}, #expected 2:1
                {0:0.75, 1:0.25}, #expected 3:1
                {0:0.8, 1:0.2}]   #expected 4:1

def main():

    # display progress
    t_start = get_start_time()

    # identify / create directories
    dir_output = f"{PROJECT_PATH}/data/results"
    if not os.path.exists(dir_output): 
        os.makedirs(f"{dir_output}")

    # load single trial spectral parameter results (pipeline step 7)
    results = load_params(f"{PROJECT_PATH}/data/results/psd_trial_params.csv")
    results = results.loc[results['ap_mode'] == AP_MODE]
    results['memory'] = results['memory'].map({'hit': 1, 'miss': 0})


    # run logistic regression for each class weight
    for ii, class_weight in enumerate(class_weights):
        # create dataframe to store results (one row for each channel-material pair)
        df = results.groupby(['patient', 'chan_idx', 'material']).count().reset_index()
        df = df[['patient', 'chan_idx', 'material']]
        df['score'] = np.nan 

        # run logistic regression for all channels in each trial condition
        features_x = ['exponent_diff', 'alpha_adj_diff', 'gamma_adj_diff', 
                    'alpha_diff', 'gamma_diff']
        feature_y = 'memory'
        for _, patient in enumerate(PATIENTS):
            print(f"patient {patient}")
            channels = results.loc[results['patient']==patient, 'chan_idx'].unique()
            for channel in channels:
                for material in ['words', 'faces']:
                    try: # some patients/channels don't have data

                        # get data for this patient/channel/material
                        res_i = results.loc[(results['patient']==patient) & \
                                            (results['chan_idx']==channel) & \
                                            (results['material']==material)]
                        
                        # match number of hits and misses
                        # n_trials = res_i['memory'].groupby(res_i['memory']).count().min()
                        # res_i = res_i.loc[res_i['memory'].isin([0, 1])].groupby('memory').head(n_trials)

                        # run logistic regression
                        _, score = run_regression(res_i, features_x, feature_y,
                                                  class_weight)
                        
                        # store results
                        df_index = (df['patient']==patient) & \
                                    (df['chan_idx']==channel) & \
                                    (df['material']==material)
                        df.loc[df_index, 'score'] = score

                    except:
                        pass

        # save results
        df.to_csv(f"{dir_output}/regression_scores_run{ii}.csv",
                index=False)

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


def load_params(fname):
    """
    Load single trial spectral parameter results i.e. pipeline step 7.
    Compute the difference in spectral parameters between baseline and
    encoding.    
    """

    df = pd.read_csv(fname)
    features = ["exponent", "alpha", "alpha_adj", "gamma", "gamma_adj"]
    pivot_index = ["patient", "material", "memory", "chan_idx", "trial", 
                   "ap_mode"] 
    df = df.pivot_table(index=pivot_index, columns="epoch", values=features)
    for feature in features:
        df[(feature, "diff")] = df[(feature, 'post')] - df[(feature, 'pre')]
        df.drop(columns=[(feature, 'pre'), (feature, 'post')], inplace=True)
    df.columns = [f"{feature}_{epoch}" for feature, epoch in df.columns]
    df.reset_index(inplace=True)
    
    return df


def run_regression(results, features_x, feature_y, class_weight):
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
    class_weight : str or dict
        Class weight to use for logistic regression.

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
    clf = LogisticRegression(random_state=0, class_weight=class_weight)
    scores = cross_val_score(clf, X, y, cv=5)
    score = scores.mean()

    return clf, score


if __name__ == "__main__":
    main()
