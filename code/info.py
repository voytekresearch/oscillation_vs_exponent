"""
Dataset details, analysis settings and hyperparameters
"""

# Imports
import numpy as np

# Dataset info
PATIENTS = ['pat02','pat04','pat05','pat08','pat10','pat11','pat15','pat16',
            'pat17','pat19','pat20','pat21','pat22']
FS = 512 # iEEG sampling frequency
TMIN = -1.5 # epoch start time

# Analysis settings
N_JOBS = -1 # number of jobs for parallel processing
ALPHA_RANGE = [7, 13] # alpha frequency range

# SpecParam hyperparameters
FREQ_RANGE = [4, 100] # frequency range to fit
SPEC_PARAM_SETTINGS = {
    'peak_width_limits' :   [2, np.inf], # default : (0.5, 12.0) - recommends at least frequency resolution * 2
    'min_peak_height'   :   0, # default : 0
    'max_n_peaks'       :   4, # default : inf
    'peak_threshold'    :   3} # default : 2.0