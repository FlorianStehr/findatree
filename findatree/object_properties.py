from typing import List
from typing import Tuple
import numpy as np

import skimage.measure as measure


#%%
def labels_idx(labels_in: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    '''
    Return indices of all labels as list.

    Parameters:
    -----------
    labels_in: np.ndarray
        Integer labeled connected components of an image (see skimage.measure.labels).
    
    Returns:
    --------
    List of ``len(labels)``. Each entry corresponds to ``Tuple[np.ndarray,np.ndarray]`` containing (coordinate) indices of all pixels belonging to unique label in original image.

    '''
    shape = labels_in.shape # Original shape of labels, corresponds to shape of image
    labels = labels_in.flatten() # Flatten

    # Get number of unique counts in labels plus the sort-index of the labels to reconstruct indices of each label.
    labels_len = np.unique(labels, return_counts=True)[1]
    labels_sortidx = np.argsort(labels)

    # Now loop through each label and get indice
    labels_idx = []
    i0 = 0
    for l in labels_len:
        i1 = i0 + l
        label_idx = labels_sortidx[i0:i1]
        i0 = i1

        # Go from flattened index to coordinate index
        label_idx = np.unravel_index(label_idx, shape) 
        
        labels_idx.append(label_idx)

    labels_idx = labels_idx[1:] # Remove index belonging to background (i.e. labels==0)

    return labels_idx


#%%
def extract_props(labels, channels, px_width):

    shape = labels.shape
    c_names = [key for key in channels]
    c_n = len(c_names)

    image = np.zeros((shape[0], shape[1], c_n))
    for i, key in enumerate(c_names):
        image[:,:,i] = channels[key]

    # Use skimage to get object properties
    props = measure.regionprops(labels,image)
    
    # Get label ID
    labels = np.array([[prop['label']] for prop in props], dtype=np.float32)

    # Get area props
    area_names = [
        'area',
        'area_convex',
        'area_filled',
    ]
    areas = np.array([[prop[area_name] for area_name in area_names] for prop in props], dtype=np.float32)
    areas = areas * px_width**2 # Unit conversion of areas to m**2

    # Get distance props
    distance_names = [
        'axis_major_length',
        'axis_minor_length',
        'equivalent_diameter_area',
        'perimeter',
        'perimeter_crofton',
        'feret_diameter_max',
    ]
    distances = np.array([[prop[distance_name] for distance_name in distance_names] for prop in props], dtype=np.float32)
    distances = distances * px_width # Unit conversion of areas to m

    # Get ratio props
    ratio_names = [
        'eccentricity',
        'extent',
        'solidity',
    ]
    ratios = np.array([[prop[ratio_name] for ratio_name in ratio_names] for prop in props], dtype=np.float32)

    # Join props that we got so far
    props_out = np.hstack((labels, areas, distances, ratios))
    names_out = ['label'] + area_names + distance_names + ratio_names

    
    # Get intensity props
    intensity_names = [
        'intensity_min',
        'intensity_mean',
        'intensity_max',
    ]
    intensities_all_names = []
    
    for i, key in enumerate(c_names):
        
        # Get intensities for one channel
        intensities = np.array([[prop[intensity_name][i] for intensity_name in intensity_names] for prop in props], dtype=np.float32)
        
        # Add intensities to so far joint props
        props_out = np.hstack((props_out, intensities))
        
        # Add names to so far joint names
        names_out = names_out + [intensity_name[10:] + '_' + key for intensity_name in intensity_names]

    
    return props_out, names_out