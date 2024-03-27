"""
This script plots the average power spectra for channels identified in
scripts.3_id_modulated_channels.py as being task-modulated.

"""

# Imports - standard
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from info import MATERIALS, MEMORY
from plots import plot_spectra_2conditions
from settings import COLORS

# settings
plt.style.use('mplstyle/default.mplstyle')
FIGSIZE = [3, 2]


def main():

    # load results of step 3
    fname = f"{PROJECT_PATH}/data/results/ieeg_modulated_channels.csv"
    stats = pd.read_csv(fname, index_col=0)

    # make directory for output figures
    dir_fig = f"{PROJECT_PATH}/figures/group_spectra"
    if not os.path.exists(f"{dir_fig}"): 
        os.makedirs(f"{dir_fig}")

    for material in MATERIALS:
        for memory in MEMORY:
            for feature in ['alpha', 'gamma', 'all', 'any']:
                plot_group_spectra(stats, material, memory, feature, dir_fig)


def plot_group_spectra(stats, material, memory, feature, dir):
    # load data
    fname = f"{PROJECT_PATH}/data/ieeg_spectral_results/psd_{material}_{memory}_XXXstim.npz"
    data_pre = np.load(fname.replace("XXX", "pre"))
    data_post = np.load(fname.replace("XXX", "post"))
    psd_pre = data_pre['spectra'][stats[f"sig_{feature}"]]
    psd_post = data_post['spectra'][stats[f"sig_{feature}"]]
    freq = data_pre['freq']

    # plot
    if material=='words':
        colors = [COLORS['light_brown'], COLORS['brown']]
    elif material=='faces':
        colors = [COLORS['light_blue'], COLORS['blue']]

    # title = f"{material} - {memory} - {feature}"
    title = f"{material[0].upper()}{material[1:]}-encoding"
    fname = f"{material}_{memory}_{feature}.png"
    f_mask = np.logical_and(freq>1, freq<100)
    _, ax = plt.subplots(figsize=FIGSIZE)
    plot_spectra_2conditions(psd_pre[:, f_mask], psd_post[:, f_mask], 
                             freq[f_mask], shade_sem=True, color=colors,
                             title=title, fname=f"{dir}/{fname}", ax=ax)

        
if __name__ == "__main__":
    main()
