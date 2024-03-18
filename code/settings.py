"""
Analysis settings and hyperparameters
"""

# Imports
import numpy as np

# General analysis settings
N_JOBS = -1 # number of jobs for parallel processing
EPOCH_TIMES = np.array([[-1.0, 1.0],    # epoch
                       [-1.0, 0.0],    # pre-stim
                       [0.0, 1.0]])    # post-stim
EPOCH_LABELS = np.array(['epoch', 'prestim', 'poststim'])

# Oscillation band settings
ALPHA_RANGE = [7, 13] # alpha frequency range
GAMMA_BANGE = [50, 90] # gamma frequency range
BANDS = {'alpha' : ALPHA_RANGE, 'gamma' : GAMMA_BANGE} # oscillation bands
FELLNER_BANDS = {'theta' : [2, 5], 'alpha' : [8, 20], 'gamma' : [50, 90]} # oscillation bands used in Fellner et al. 2016

# power analysis settings
WINDOW = 0.3 # time window for spectral decomposition
TFR_METHOD = 'multitaper' # 'morlet' or 'multitaper'
N_TFR_SAMPLES = 2**8 # number of samples after downsampling TFR

# SpecParam hyperparameters
FREQ_RANGE = [4, 100] # frequency range to fit
SPEC_PARAM_SETTINGS = {
    'peak_width_limits' :   [2, np.inf], # default : (0.5, 12.0) - recommends at least frequency resolution * 2
    'min_peak_height'   :   0, # default : 0
    'max_n_peaks'       :   4, # default : inf
    'peak_threshold'    :   3} # default : 2.0
AP_MODE = 'knee'

# Plotting
COLORS = {
    "brown"         :   np.array([166,97,26]) / 255,
    "light_brown"   :   np.array([223,194,125]) / 255,
    "blue"          :   np.array([1,133,113]) / 255,
    "light_blue"    :   np.array([128,205,193]) / 255,
}
