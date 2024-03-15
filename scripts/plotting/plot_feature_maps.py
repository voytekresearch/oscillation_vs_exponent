"""
Plot results of scripts/pipeline/step4_spectral_parameterization.py as brainmap.

"""

# Imports - standard
import os
import pandas as pd
import matplotlib.pyplot as plt
from neuromaps.datasets import fetch_atlas
import nibabel as nib

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from map_utils import create_brain_map, plot_glass_brain_map
from utils import get_start_time, print_time_elapsed

# settings
FEATURES = ['exponent', 'alpha_adj', 'gamma_adj'] # spectral features of interest
plt.style.use('mplstyle/default.mplstyle')

def main():
    # display progress
    t_start = get_start_time()

    # identify / create directories
    dir_data = f"{PROJECT_PATH}/data/neuromaps/feature_map_nifti"
    dir_figs = f"{PROJECT_PATH}/figures/neuromaps/feature_maps"
    for dir_output in [dir_data, dir_figs]:
        if not os.path.exists(dir_output): 
            os.makedirs(f"{dir_output}")    

    # load template MNI brain
    mni152_atlas = fetch_atlas('MNI152', '1mm')
    mni152_template = nib.load(mni152_atlas['6Asym_brainmask'])

    # load electrode locations
    fname = f"{PROJECT_PATH}/data/ieeg_metadata/ieeg_channel_info.csv"
    df_elec = pd.read_csv(fname, index_col=0)

    # loop through conditions
    for material in ['words', 'faces']:
        # display progress
        print("\n\n==================================")
        print(f'Analyzing condition:\t{material}')

        # load spec param results
        fname = f"{PROJECT_PATH}/data/results/spectral_parameters.csv"
        df_in = pd.read_csv(fname, index_col=0)

        # get data for condition
        df_cond = df_in.loc[(df_in['material'] == material) & \
                            (df_in['memory']=='hit')].reset_index(drop=True)
        
        # merge with electrode info
        df = df_cond.merge(df_elec)

        # pivot table - so evoked change in features can be computed
        df = df.pivot(index=['patient', 'chan_idx'], columns='epoch', 
                            values=FEATURES).reset_index()
        
        # analyze each feature
        for feature in FEATURES:
            # display progress
            print(f'Analyzing feature:\t{feature}')
            
            # compute evoked change in each feature
            post = df.loc[:, pd.IndexSlice[feature, 'post']]
            pre = df.loc[:, pd.IndexSlice[feature, 'pre']]
            df[f'diff_{feature}'] = post - pre

            # create brain map for feature
            brain_map = create_brain_map(df, f'diff_{feature}', 
                                         mni152_template)
            
            # convert to Nifti image, save, and plot
            image = nib.Nifti1Image(brain_map, mni152_template.affine)
            fname = f"{feature}_{material}"
            nib.save(image, f"{dir_data}/{fname}.nii.gz")
            plot_glass_brain_map(brain_map, mni152_template.affine, True,
                                 fname_out=f"{dir_figs}/{fname}.png")

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


if __name__ == "__main__":
    main()
