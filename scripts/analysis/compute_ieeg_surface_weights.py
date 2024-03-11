"""
THis script computes the distance between each electrode ih the dataset and
a template brain surface map.

"""


import os
import numpy as np
import pandas as pd
from neuromaps.datasets import fetch_atlas
import nibabel as nib

import sys
sys.path.append("code")
from paths import PROJECT_PATH
from info import PATIENTS
from map_utils import apply_affine, compute_distances, compute_weights
from utils import get_start_time, print_time_elapsed

# settings
SPREAD = 5 # weight is 50% at a distance of 'spread' voxels


def main():
        
    # display progress
    t_start = get_start_time()

    # identify / create directories
    dir_output = f"{PROJECT_PATH}/data/neuromaps/mni_surface_weights"
    if not os.path.exists(dir_output): 
        os.makedirs(f"{dir_output}")

    # load template MNI brain
    mni152_atlas = fetch_atlas('MNI152', '1mm')
    mni152_template = nib.load(mni152_atlas['6Asym_brainmask'])
    output_grid = np.array(np.where(np.asarray(mni152_template.get_fdata())>0)).T # coordinates of brain surface

    # load electrode locations
    df = pd.read_csv(f"{PROJECT_PATH}/data/ieeg_metadata/ieeg_channel_info.csv", index_col=0)

    # compute weight matrix for each patients and save to file (~1GB per patient)
    for patient in PATIENTS:

        # display progress
        t_start_p = get_start_time()
        print(f"Analyzing patient:\t{patient}")

        # apply affine transform to convert iEEG coordinates to brain map indices
        df_patient = df.loc[df['patient']==patient]
        input_grid = np.array([apply_affine(mni152_template.affine, row[['pos_x','pos_y','pos_z']].values, False) for _, row in df_patient.iterrows()])    
        
        # compute distance between iEEG locations and template brain surface
        distance = compute_distances(input_grid, output_grid)
        
        # compute weight matrix and save to file
        weights = compute_weights(distance, spread=SPREAD)
        np.save(f"{dir_output}/{patient}.npy", weights)

        # display progress
        print_time_elapsed(t_start_p)

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


if __name__ == "__main__":
    main()
