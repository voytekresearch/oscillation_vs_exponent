"""
Utility functions for working with 3D brain maps. These functions are
adapted from: Gao et al., 2021 (https://github.com/rdgao/field-echos).

"""

# imports
import numpy as np

def apply_affine(affine, coordinates, forward=True):
    """
    Apply forward (index to MNI coor) or reverse (MNI to index) affine transformation.

    Adapted from: https://github.com/rdgao/field-echos
    """
    
    m_aff, m_trsl = affine[:3,:3], affine[:3,-1]
    if forward:
        # index to MNI coordinate
        return np.dot(m_aff,coordinates)+m_trsl
    else:
        # MNI coordinate to index
        return np.dot(np.linalg.inv(m_aff),coordinates-m_trsl)
    

def project_feature(feature, input_grid, output_grid, spread=5):
    """
    This function projects a feature matrix from one grid to another. The
    projection is computed by first computing the Euclidean distance between
    each coordinate in input_grid and each coordinate in output_grid. A
    Gaussian weighting function is then applied to the distance matrix to
    determine the contribution of each input coordinate to each output
    coordinate. The Gaussian weighting function is parameterized such that the
    weight is 50% at a distance of 'spread' voxels. The weighted feature matrix
    is then computed as the dot product of the feature matrix and the weight
    matrix.

    adapted from: https://github.com/rdgao/field-echos
    """

    # compute distance between each voxel in input_ and ouuput_grid
    dist = compute_distances(input_grid, output_grid)

    # compute weight matrix
    weights = compute_weights(dist, spread=spread)

    # apply weights
    feat_weighted = apply_feature_weights(feature, weights)

    return feat_weighted


def weight_feature(feature, distance, spread=5):
    """
    This function can be used to project a feature matrix from one grid to 
    another. This function takes in the Euclidean distance between
    each coordinate in an input grid and each coordinate in an output grid. A
    Gaussian weighting function is then applied to the distance matrix to
    determine the contribution of each input coordinate to each output
    coordinate. The Gaussian weighting function is parameterized such that the
    weight is 50% at a distance of 'spread' voxels. The weighted feature matrix
    is then computed as the dot product of the feature matrix and the weight
    matrix.

    adapted from: https://github.com/rdgao/field-echos
    """

    # compute weight matrix
    weights = compute_weights(distance, spread=spread)

    # apply weights
    feature_weighted = apply_feature_weights(feature, weights)

    return feature_weighted


def compute_distances(input_grid, output_grid):
    """
    This function computes the Euclidean distance between each coordinate in
    input_grid and each coordinate in output_grid.  

    adapted from: https://github.com/rdgao/field-echos
    """
    
    dist = np.zeros((input_grid.shape[0], output_grid.shape[0]))
    for ig in range(input_grid.shape[0]):
        for og in range(output_grid.shape[0]):
            dist[ig, og] = np.linalg.norm(output_grid[og]-input_grid[ig])

    return dist


def compute_weights(dist, spread=5):
    """
    This function applies a Gaussian weighting function to the distance matrix
    to determine the contribution of each input coordinate to each output
    coordinate. The Gaussian weighting function is parameterized such that the
    weight is 50% at a distance of 'spread' voxels.

    adapted from: https://github.com/rdgao/field-echos
    """

    # set smoothing parameter: Gaussian is at 50% when d voxels away
    d_alpha = spread/(-np.log(0.5))**0.5

    # create the weight matrix from input to output projection based on Gaussian weighting of Euclidean distance
    weights = np.exp(-dist**2/d_alpha**2)

    return weights


def apply_feature_weights(feature, weights):
    """
    This function applies a weighting function to a feature matrix.

    adapted from: https://github.com/rdgao/field-echos

    """

    # apply weights
    feat_weighted = np.dot(feature, weights)/weights.sum(0)

    return feat_weighted
    

def compute_weighted_average(df_feature, df_W, w_thresh=0.5, axis=0, method='weighted'):
    """
    Compute weighted average of feature matrix.

    Adapted from: https://github.com/rdgao/field-echos
    """

    if method=='weighted':
        # method 1: weighted average of all parcels
        #    this makes the most sense? weights all parcels, and the non-confident ones are already downweighted
        return (df_feature*df_W).sum(axis=axis)/df_W.sum(axis=axis)

    elif method=='thresh_weighted':
        # method 2: weighted average of suprathreshold parcels
        #    this will approach method 1 as w_thresh approaches 0
        thresh_mat = df_W>=w_thresh
        return (df_feature*df_W)[thresh_mat].sum(axis=axis)/df_W[thresh_mat].sum(axis=axis)

    elif method=='thresh_mean':
        # method 3: simple average of suprathreshold parcels
        #    not sure if it makes sense to weight all suprathreshold parcels equally
        thresh_mat = df_W>=w_thresh
        return np.nanmean((df_feature*df_W)[thresh_mat],axis=axis)
    

def surface_coordinates_to_map(surface_coords, surface_values, template_shape):
    """
    This function converts a set of surface coordinates and values to a 3D
    brain map.

    Adapted from: https://github.com/rdgao/field-echos
    """

    brain_map = np.zeros(template_shape)
    for ii, coord in enumerate(surface_coords):
        brain_map[coord[0], coord[1], coord[2]] = surface_values[ii]

    return brain_map