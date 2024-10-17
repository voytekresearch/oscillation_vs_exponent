"""
Justify aperiodic mode
"""

# Imports - standard
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from specparam import SpectralModel, SpectralGroupModel

# Imports - custom
import sys
sys.path.append(f"code")
from paths import PROJECT_PATH
from settings import SPEC_PARAM_SETTINGS, RGB, WIDTH, FREQ_RANGE
from specparam_utils import compute_adj_r2
from utils import get_start_time, print_time_elapsed
from plots import beautify_ax

# settings
plt.style.use('mplstyle/nature_neuro.mplstyle')
REMOVE_OUTLIERS = True


def main():

    # display progress
    t_start = get_start_time()

    # identify / create directories
    dir_output = f"{PROJECT_PATH}/data/results"
    dir_fig = f"{PROJECT_PATH}/figures/main_figures"
    for dir in [dir_output, dir_fig]:
        if not os.path.exists(dir): 
            os.makedirs(f"{dir}")
        
    # load r-squared values and compute diff ===================================
    r2_knee = []
    r2_fixed = []

    # loop through conditions
    for material in ['words','faces']:
        for memory in ['hit','miss']:
            for window in ['pre', 'post']:
                # load r-squared values for 'knee' model
                fg_k = SpectralGroupModel()
                fname = f"psd_{material}_{memory}_{window}stim_params_knee"
                fg_k.load(f'{PROJECT_PATH}/data/ieeg_psd_param/{fname}')
                r2_k = compute_adj_r2(fg_k)
                r2_knee.append(r2_k)

                # load r-squared values for 'fixed' model
                fg_f = SpectralGroupModel()
                fname = f"psd_{material}_{memory}_{window}stim_params_fixed"
                fg_f.load(f'{PROJECT_PATH}/data/ieeg_psd_param/{fname}')
                r2_f = compute_adj_r2(fg_f)
                r2_fixed.append(r2_f)

    # concatenate
    r2_knee = np.concatenate(r2_knee)
    r2_fixed = np.concatenate(r2_fixed)

    # compute difference in r-squared betwen models
    r2_diff = r2_knee - r2_fixed

    # load all PSDs and fit grand average ======================================
    dir_input = f'{PROJECT_PATH}/data/ieeg_spectral_results'
    files = [f for f in os.listdir(dir_input) if \
             (('psd' in f) and ('epoch' in f))]

    # init
    temp = np.load(f"{dir_input}/{files[0]}")
    spectra = np.zeros([len(files), temp['spectra'].shape[0], 
                        temp['spectra'].shape[1]])
    freq = temp['freq']
    del temp

    # load spectra for each condition
    for i_file, fname_in in enumerate(files):
        data_in = np.load(f"{dir_input}/{fname_in}")
        spectra[i_file] = data_in['spectra']

    # crop to fitting freq range
    freq_idx = np.where((freq>=FREQ_RANGE[0]) & (freq<=FREQ_RANGE[1]))[0]
    freq = freq[freq_idx]
    spectra = spectra[:,:,freq_idx]
        
    # average over channels and compute grand verage
    spectra_cond = np.nanmean(spectra, axis=1)
    psd = np.nanmean(spectra_cond, axis=0)

    # fit 'knee'
    fg_k = SpectralModel(aperiodic_mode='knee', **SPEC_PARAM_SETTINGS)
    fg_k.fit(freq, psd)
    r2_k = compute_adj_r2(fg_k)

    # fit 'fixed'
    fg_f = SpectralModel(aperiodic_mode='fixed', **SPEC_PARAM_SETTINGS)
    fg_f.fit(freq, psd)
    r2_f = compute_adj_r2(fg_f)

    # plot =====================================================================
    # create figure and gridspec
    fig = plt.figure(figsize=[WIDTH['1col'], WIDTH['1col']])
    spec = gridspec.GridSpec(figure=fig, ncols=2, nrows=2, width_ratios=[1, 1], 
                             height_ratios=[1, 1.5])
    ax1 = fig.add_subplot(spec[0,0])
    ax2 = fig.add_subplot(spec[0,1])

    # create nested gridspec for grand average PSD
    spec2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=spec[1,:],
                                            width_ratios=[0.17, 0.66, 0.17])
    ax3 = fig.add_subplot(spec2[1])

    # plot subplot 1 - violin --------------------------------------------------
    # remove nan
    r2_knee_ = r2_knee[~np.isnan(r2_knee)]
    r2_fixed_ = r2_fixed[~np.isnan(r2_fixed)]

    # remove outliers
    if REMOVE_OUTLIERS:
        outlier_thresh_knee = np.nanmean(r2_knee) - 5 * np.nanstd(r2_knee)
        outlier_thresh_fixed = np.nanmean(r2_fixed) - 5 * np.nanstd(r2_fixed)
        print(f"Removed {len(r2_knee_[r2_knee_<=outlier_thresh_knee])} outliers from knee model")
        print(f"Removed {len(r2_fixed_[r2_fixed_<=outlier_thresh_fixed])} outliers from fixed model")

    # plot
    ax1.set_title('Single-channel fits')
    if REMOVE_OUTLIERS:
        sns.violinplot(data=[r2_knee_[r2_knee_>outlier_thresh_knee], 
                            r2_fixed_[r2_fixed_>outlier_thresh_fixed]], ax=ax1,
                            palette=[RGB[2], RGB[1]])
    else:
        sns.violinplot(data=[r2_knee_, r2_fixed_], ax=ax1,
                        palette=[RGB[2], RGB[1]])
    ax1.set(ylabel="adjusted $R^2$")
    ax1.set_xticks([0, 1], ['knee', 'fixed'], fontsize=6) # CHECK - match to xlabel

    # plot subplot 2 - histogram -----------------------------------------------
    ax2.hist(r2_diff, bins=60, alpha=0.5, label='knee - fixed', color='grey')
    ax2.set(xlabel="$R^2_{knee} - R^2_{fixed}$", ylabel="count")
    ax2.axvline(0, color='k', linestyle='--')
    ax2.set_title('Difference in $R^2$')

    # plot subplot 3 - GA PSD --------------------------------------------------
    ax3.plot(freq, psd, label='grand average', color='k')
    ax3.plot(fg_k.freqs, 10**fg_k._ap_fit, color=RGB[2], linewidth=3, alpha=0.5, 
            label='knee model')
    ax3.plot(fg_f.freqs,10**fg_f._ap_fit, color=RGB[1], linewidth=3, alpha=0.5, 
            label='fixed model')
    ax3.set(xlabel="frequency (Hz)", ylabel="power ($\u03BCV^2$/Hz)")
    ax3.set_title('Grand average PSD')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xticks([10, 100], ['10', '100'])
    ax3.legend(loc='lower left')

    # beautify axes
    for ax in [ax1, ax2, ax3]:
        beautify_ax(ax)

    # save fig
    fig.savefig(f"{dir_fig}/figure_4-1.png")
    fig.savefig(f"{dir_fig}/figure_4-1")

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


if __name__ == "__main__":
    main()
