from typing import List, Tuple, Dict
import numpy as np

import skimage.measure

from numpy.lib.recfunctions import unstructured_to_structured
from tqdm import tqdm


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
def prop_to_intensitycoords(prop, channels, brightness_channel):
    
    ####################### Define image-indices and channel values of the segment

    # Channel names
    channel_names = [key for key in channels] 

    # Indices of the label, in image coordinates
    idxs = prop['coords']
    idxs = (idxs[:,0], idxs[:,1])

    # Prepare flat array of values of each channel. Row corresponds to channel.
    channels_vals = np.zeros((len(channel_names), len(idxs[0])), dtype=np.float32)
    for i, key in enumerate(channels):
        img = channels[key]
        channels_vals[i,:] = img[idxs]


    ####################### Define brightest indices and values of the segment

    # Which channel index corresponds to brightness channel, in channels_vals coordinates, eg. which row?
    channels_brightness_idx = channel_names.index(brightness_channel)
    
    # Get brightness values
    brightness_vals = channels_vals[channels_brightness_idx, :]
    
    # Set the threshold for bright pixels to upper 75 percentile of brightness values
    brightness_thresh = np.percentile(brightness_vals, 75)

    # Get indices where brightness values are above threshold, in channels_vals coordinates.
    brightness_upper_idxs = np.where(brightness_vals > brightness_thresh)[0]
    
    # These are the indices of the brightest pixels, in image coordinates.
    idxs_brightest = (idxs[0][brightness_upper_idxs], idxs[1][brightness_upper_idxs])
    
    # These are the brightest values of all channels
    channels_vals_brightest = channels_vals[:, brightness_upper_idxs]


    ####################### Feature extraction

    # Init data and names
    data = []
    names = []

    ############# All pixels in segment
    
    # Add total number of pixels
    names.extend(['n_px'])
    data.append(len(idxs[0]))

    # Add minimum value of each channel
    names.extend(['min_' + name for name in channel_names])
    data.extend( list( np.nanmin(channels_vals, axis=1)[:] ) )

    # Add maximum value of each channel
    names.extend(['max_' + name for name in channel_names])
    data.extend( list( np.nanmax(channels_vals, axis=1)[:] ) )

    # Add mean value of each channel
    names.extend(['mean_' + name for name in channel_names])
    data.extend( list( np.nanmean(channels_vals, axis=1)[:] ) )

    # Add std value of each channel
    names.extend(['std_' + name for name in channel_names])
    data.extend( list( np.nanstd(channels_vals, axis=1)[:] ) )

    # Add median value of each channel
    names.extend(['median_' + name for name in channel_names])
    data.extend( list( np.nanmedian(channels_vals, axis=1)[:] ) )


    ############# Brightest pixels in segment
    
    # Add number of brightest pixels
    names.extend(['n_px_brightest'])
    data.append(len(idxs_brightest[0]))
    
    # Add mean value of each channel over brightest pixels
    names.extend(['mean_brightest_' + name for name in channel_names])
    data.extend( list( np.nanmean(channels_vals_brightest, axis=1)[:] ) )

    # for i, key in enumerate(channel_names):
        
        # Get values of one channel
        # channel_vals = channels_vals[i, :]

        # names.extend(['min_' + key])
        # data.append(np.min(channel_vals))

        # names.extend(['max_' + key])
        # data.append(np.max(channel_vals))

        # names.extend(['mean_' + key])
        # data.append(np.median(channel_vals))
        
        # names.extend(['std_' + key])
        # data.append(np.std(channel_vals))

        # names.extend(['median_' + key])
        # data.append(np.mean(channel_vals))

        # names.extend(['perc25_' + key])
        # data.append(np.percentile(channel_vals, 25))

        # names.extend(['perc75_' + key])
        # data.append(np.percentile(channel_vals, 75))
        
        # names.extend(['x_max_' + key])
        # data.append(idxs[1][np.argmax(channel_vals)])

        # names.extend(['y_max_' + key])
        # data.append(idxs[0][np.argmax(channel_vals)])

        # names.extend(['x_com_' + key])
        # data.append(np.sum(idxs[1] * channel_vals) / np.sum(channel_vals))

        # names.extend(['y_com_' + key])
        # data.append(np.sum(idxs[0] * channel_vals) / np.sum(channel_vals))


        # names.extend(['mean_darkpx_' + key])
        # data.append(np.mean(channel_vals[light_lower_idxs]))

        # names.extend(['mean_brightpx_' + key])
        # data.append(np.mean(channel_vals[light_upper_idxs]))


    data = np.array(data, dtype=np.float32)

    return data, names


#%%
def prop_to_allfeatures(prop, channels, px_width, brightness_channel):
    
    # Get label
    label_name = ['id']                 # We will store the crown identifier as `id` ...
    label = np.array([ prop['label'] ]) # ... but in skimage it's called `label`

    # Get areas
    areas, area_names = prop_to_areas(prop)
    areas = areas * px_width**2 # Unit conversion from px to [m**2]

    # Get distances
    distances, distance_names = prop_to_distances(prop)
    distances = distances * px_width # Unit conversion from px to [m]

    # Get ratios
    ratios, ratio_names = prop_to_ratios(prop)

    # Get intensities and coordinates
    intcoords, intcoord_names = prop_to_intensitycoords(
        prop,
        channels, 
        brightness_channel=brightness_channel,
        )

    # Concatenate all props and corresponding names
    features = np.concatenate(
        (label,
        areas,
        distances,
        ratios,
        intcoords,
        ),
    )
    names = label_name + area_names + distance_names + ratio_names + intcoord_names

    return features, names

#%%

def labelimage_extract_features(
    labelimg,
    channels_in,
    params_channels,
    include_labels = None,
    brighntess_channel = 'light',
    ):

    # Copy input channels
    channels = channels_in.copy()

    # Use skimage to get object properties
    props = skimage.measure.regionprops(labelimg, None)
    
    # Only compute props of labels included in include_labels
    if include_labels is not None:
        props_include = [prop for prop in props if prop['label'] in include_labels]
    else:
        props_include = props

    # Loop through all labels and extract properties as np.ndarray
    for i, prop in enumerate(tqdm(props_include)):
        
        if i == 0: # First assignment
            
            features, names  = prop_to_allfeatures(
                prop,
                channels,
                params_channels['px_width'],
                brightness_channel = brighntess_channel,
                )

        else:
            
            features_i, names = prop_to_allfeatures(
                prop,
                channels,
                params_channels['px_width'],
                brightness_channel = brighntess_channel,
                )
            
            features = np.vstack((features, features_i))


    # Prepare dtype for conversion of features to structured numpy array
    dtypes = ['<f4' for name in names]
    dtypes[0] = '<u4'
    dtypes = [(name, dtype) for name, dtype in zip(names, dtypes)]

    # Convert features to structures dtype
    features = unstructured_to_structured(features, dtype=np.dtype(dtypes))


    return features