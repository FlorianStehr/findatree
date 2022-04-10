from typing import List
from typing import Tuple
import numpy as np
from tqdm import tqdm

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
def prop_to_areas(prop):
    
    area_names = [
        'area',
        'area_convex',
        'area_filled',
    ]
    areas = np.array([prop[area_name] for area_name in area_names])
    return areas, area_names


#%%
def prop_to_distances(prop):
    distance_names = [
        'axis_major_length',
        'axis_minor_length',
        'equivalent_diameter_area',
        'perimeter',
        'perimeter_crofton',
        'feret_diameter_max',
    ]
    distances = np.array([prop[distance_name] for distance_name in distance_names], dtype=np.float32)
    return distances, distance_names

#%%
def prop_to_ratios(prop):
    ratio_names = [
        'eccentricity',
        'extent',
        'solidity',
    ]
    ratios = np.array([prop[ratio_name] for ratio_name in ratio_names], dtype=np.float32)
    return ratios, ratio_names


#%%
def prop_to_intensitycoords(prop, channels):
    
    c_names = [key for key in channels]  # Channel names
    c_shape = channels[c_names[0]]  # Shape of image

    # Indices of the label in image coordinates
    idx = prop['coords']
    idx = (idx[:,0], idx[:,1])

    # Prepare flat array of values of each channel
    c_vals = np.zeros((len(c_names), len(idx[0])), dtype=np.float32)
    for i, key in enumerate(channels):
        img = channels[key]
        c_vals[i,:] = img[idx]

    # Which channel index corresponds to lightness channel
    c_l_idx = c_names.index('l')
    # Get lighntess values
    l_vals = c_vals[c_l_idx, :]
    # Set the threshold to upper 50 percentile of lightness values
    l_thresh = np.percentile(l_vals, 50)
    l_max = np.max(l_vals)
    # if l_thresh == l_max: # Exception for totally homogeneos and/or small number of lighntess values
    #     l_thresh = np.min(l_vals)

    # Get indices where lighntess values are above threshold
    l_upper_idx = np.where(l_vals > l_thresh)[0]
    l_lower_idx = np.where(l_vals <= l_thresh)[0]
    idx_bright = (idx[0][l_upper_idx], idx[1][l_upper_idx])

    # Init data
    data = []
    names = []

    # Add number of bright pixels
    names.extend(['n_px_bright'])
    data.append(len(idx_bright[0]))

    for i, key in enumerate(c_names):
        # Get values of one channel
        c_val = c_vals[i, :]

        names.extend(['min_' + key])
        data.append(np.min(c_val))

        names.extend(['max_' + key])
        data.append(np.max(c_val))

        names.extend(['mean_' + key])
        data.append(np.median(c_val))
        
        names.extend(['std_' + key])
        data.append(np.std(c_val))

        names.extend(['median_' + key])
        data.append(np.mean(c_val))

        names.extend(['perc25_' + key])
        data.append(np.percentile(c_val, 25))

        names.extend(['perc75_' + key])
        data.append(np.percentile(c_val, 75))

        names.extend(['mean_lowerl_' + key])
        data.append(np.mean(c_val[l_lower_idx]))

        names.extend(['mean_upperl_' + key])
        data.append(np.mean(c_val[l_upper_idx]))

        names.extend(['x_max_' + key])
        data.append(idx[1][np.argmax(c_val)])

        names.extend(['y_max_' + key])
        data.append(idx[0][np.argmax(c_val)])

        names.extend(['x_com_' + key])
        data.append(np.sum(idx[1] * c_val) / np.sum(c_val))

        names.extend(['y_com_' + key])
        data.append(np.sum(idx[0] * c_val) / np.sum(c_val))

        names.extend(['x_perc75_' + key])
        data.append(np.mean(idx[1][c_val > np.percentile(c_val, 75)]))

        names.extend(['y_perc75_' + key])
        data.append(np.mean(idx[0][c_val > np.percentile(c_val, 75)]))

        names.extend(['x_perc25_' + key])
        data.append(np.mean(idx[1][c_val < np.percentile(c_val, 25)]))

        names.extend(['y_perc25_' + key])
        data.append(np.mean(idx[0][c_val < np.percentile(c_val, 25)]))


    data = np.array(data, dtype=np.float32)

    return data, names


#%%
def prop_to_all(prop, channels, px_width):
    # Get label
    label_name = ['label']
    label = np.array([prop[label_name[0]]])

    # Get areas
    areas, area_names = prop_to_areas(prop)
    areas = areas * px_width**2 # Unit conversion from px to [m**2]

    # Get distances
    distances, distance_names = prop_to_distances(prop)
    distances = distances * px_width # Unit conversion from px to [m]

    # Get ratios
    ratios, ratio_names = prop_to_ratios(prop)

    # Get intensities and coordinates
    intcoords, intcoord_names = prop_to_intensitycoords(prop, channels)

    # Concatenate all props and corresponding names
    props_all = np.concatenate(
        (label,
        areas,
        distances,
        ratios,
        intcoords,
        ),
    )
    names_all = label_name + area_names + distance_names + ratio_names + intcoord_names

    return props_all, names_all

#%%

def labels_to_props_all(labels, cs, params_cs, include_labels=None):

    # Copy input channels
    channels = cs.copy()

    # Exlude RGB channel
    exclude_channels= ['RGB'] # RGB channel of shape (M,N,3) is excluded!
    for c_ex in exclude_channels:
        channels.pop(c_ex)

    # Use skimage to get object properties
    props = measure.regionprops(labels, None)
    
    # Only compute props of labels included in include_labels
    if include_labels is not None:
        props_include = [prop for prop in props if prop['label'] in include_labels]
    else:
        props_include = props

    # Loop through all labels and extract properties as np.ndarray
    for i, prop in enumerate(tqdm(props_include)):
        
        if i == 0: # First assignment
            data, names = prop_to_all(prop, channels, params_cs['px_width'])
        else:
            data_i, names = prop_to_all(prop, channels, params_cs['px_width'])
            data = np.vstack((data, data_i))


    return data, names