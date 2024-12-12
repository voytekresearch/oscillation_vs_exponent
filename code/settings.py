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
    'peak_threshold'    :   3 # default : 2.0
}
AP_MODE = 'knee'

# band power settings
BAND_POWER_METHOD = 'mean'
LOG_POWER = True

# Plotting
COLORS = {
    "brown"         :   np.array([166,97,26]) / 255,
    "light_brown"   :   np.array([223,194,125]) / 255,
    "blue"          :   np.array([1,133,113]) / 255,
    "light_blue"    :   np.array([128,205,193]) / 255,
}

BCOLORS = {
    "alpha"     :   np.array([117,112,179]) / 255, # purple
    "gamma"     :   np.array([102,166,30]) / 255, # green
    "exponent"  :   np.array([255,127,0]) / 255, # orange
}

MCOLORS = {
    "words" :   np.array([166,97,26]) / 255, # brown
    "faces" :   np.array([1,133,113]) / 255, # blue
}

RGB = (
    np.array([228,26,28]) / 255,
    np.array([77,175,74]) / 255,
    np.array([55,126,184]) / 255
)

# Journal of Neuroscience
WIDTH = {
    "1col"      :   3.34646,    # 8.5 cm
    "1.5col"    :   4.566929,   # 11.6 cm
    "2col"      :   6.929134,   # 17.6 mm
}

# Subplot labels
PANEL_FONTSIZE = 9
