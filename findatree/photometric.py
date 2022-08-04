from typing import List, Tuple, Dict
import numpy as np
import warnings
import skimage.measure
from numpy.lib.recfunctions import unstructured_to_structured
from tqdm import tqdm

import findatree.transformations as transformations


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
    channels_vals = np.zeros(
        (len(channel_names), len(idxs[0])),
        dtype=np.float32,
        )
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
    
    # Treat case of zero brightest pixels -> Assign index of maximum value
    if len(brightness_upper_idxs) == 0:
        brightness_upper_idxs = [np.argmax(brightness_vals)]
    
    # These are the indices of the brightest pixels, in image coordinates.
    idxs_brightest = (idxs[0][brightness_upper_idxs], idxs[1][brightness_upper_idxs])
    
    # These are the brightest values of all channels
    channels_vals_brightest = channels_vals[:, brightness_upper_idxs]


    ####################### Feature extraction

    # Init data and names
    data = []
    names = []

    # This is used to catch all the numpy warnings caused by numpy.nanmean() and alike 
    # when all values in one crown are NaNs. Can be ignored, because NaNs are assigned that can be filtered out later.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        ############# All pixels in segment
        
        # Add total number of pixels not including nan pixels
        names.extend(['n_px'])
        data.append( len(idxs[0]) - np.sum( np.isnan( channels_vals[0, :]) ) )

        # Add minimum value of each channel
        names.extend(['min_' + name for name in channel_names])
        data.extend( list( np.nanmin(channels_vals, axis=1) ) )

        # Add maximum value of each channel
        names.extend(['max_' + name for name in channel_names])
        data.extend( list( np.nanmax(channels_vals, axis=1) ) )

        # Add mean value of each channel
        names.extend(['mean_' + name for name in channel_names])
        data.extend( list( np.nanmean(channels_vals, axis=1) ) )

        # Add std value of each channel
        names.extend(['std_' + name for name in channel_names])
        data.extend( list( np.nanstd(channels_vals, axis=1) ) )

        # Add median value of each channel
        names.extend(['median_' + name for name in channel_names])
        data.extend( list( np.nanmedian(channels_vals, axis=1) ) )

        # Add 75 percentile value of each channel
        names.extend(['perc75_' + name for name in channel_names])
        data.extend( list( np.nanpercentile(channels_vals, 75, axis=1) ) )

        # Add 25 percentile value of each channel
        names.extend(['perc25_' + name for name in channel_names])
        data.extend( list( np.nanpercentile(channels_vals, 25, axis=1) ) )

        # Add (unweighted) center coordinate x in pixels (max)
        names.extend(['x_max_' + name for name in channel_names])
        data.extend( list( idxs[1][ np.argmax(channels_vals, axis=1) ] ) )

        # Add (unweighted) center coordinate x in pixels (mean)
        names.extend(['x_mean'])
        data.extend( [ np.mean(idxs[1] ) ] )

        # Add (unweighted) center coordinate y in pixels (mean)
        names.extend(['y_mean'])
        data.extend( [ np.mean(idxs[0] ) ] )

        # Add bounding box minimum in x in pixels
        names.extend(['x_min_bbox'])
        data.extend( [ np.min(idxs[1] ) ] )

        # Add bounding box maximum in x in pixels
        names.extend(['x_max_bbox'])
        data.extend( [ np.max(idxs[1] ) ] )

        # Add bounding box minimum in y in pixels
        names.extend(['y_min_bbox'])
        data.extend( [ np.min(idxs[0] ) ] )

        # Add bounding box maximum in y in pixels
        names.extend(['y_max_bbox'])
        data.extend( [ np.max(idxs[0] ) ] )

        ############# Brightest pixels in segment
        
        # Add number of brightest pixels not including nan pixels
        names.extend(['n_px_brightest'])
        data.append( len(idxs_brightest[0]) - np.sum( np.isnan( channels_vals_brightest[0, :]) ) )
        
        # Add minimum value of each channel of brightest pixels
        names.extend(['min_brightest_' + name for name in channel_names])
        data.extend( list( np.nanmin(channels_vals_brightest, axis=1) ) )

        # Add mean value of brightest pixels of each channel 
        names.extend(['mean_brightest_' + name for name in channel_names])
        data.extend( list( np.nanmean(channels_vals_brightest, axis=1) ) )

        # Add std value of brightest pixels of each channel
        names.extend(['std_brightest_' + name for name in channel_names])
        data.extend( list( np.nanstd(channels_vals_brightest, axis=1) ) )

        # Add median value of brightest pixels of each channel
        names.extend(['median_brightest_' + name for name in channel_names])
        data.extend( list( np.nanmedian(channels_vals_brightest, axis=1) ) )

        # Add 75 percentile value of brightest pixels of each channel
        names.extend(['perc75_brightest_' + name for name in channel_names])
        data.extend( list( np.nanpercentile(channels_vals_brightest, 75, axis=1) ) )

        # Add 25 percentile value of brightest pixels of each channel
        names.extend(['perc25_brightest_' + name for name in channel_names])
        data.extend( list( np.nanpercentile(channels_vals_brightest, 25, axis=1) ) )

        # Add (unweighted) center coordinate x of brightest pixels in pixels (mean)
        names.extend(['x_mean_brightest'])
        data.extend( [ np.mean(idxs_brightest[1] ) ] )

        # Add (unweighted) center coordinate y of brightest pixels in pixels (mean)
        names.extend(['y_mean_brightest'])
        data.extend( [ np.mean(idxs_brightest[0] ) ] )

        # Add bounding box minimum in x of brightest pixels in pixels
        names.extend(['x_min_bbox_brightest'])
        data.extend( [ np.min(idxs_brightest[1] ) ] )

        # Add bounding box maximum in x of brightest pixels in pixels
        names.extend(['x_max_bbox_brightest'])
        data.extend( [ np.max(idxs_brightest[1] ) ] )

        # Add bounding box minimum in y of brightest pixels in pixels
        names.extend(['y_min_bbox_brightest'])
        data.extend( [ np.min(idxs_brightest[0] ) ] )

        # Add bounding box maximum in y of brightest pixels in pixels
        names.extend(['y_max_bbox_brightest'])
        data.extend( [ np.max(idxs_brightest[0] ) ] )

    ############# Create final output array

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
    channels,
    params_channels,
    include_ids = None,
    brightness_channel = 'light',
    ):

    # Use skimage to get object properties
    props = skimage.measure.regionprops(labelimg, None)
    
    # Only compute props of labels included in include_labels
    if include_ids is not None:
        props_include = [prop for prop in props if prop['label'] in include_ids]
    else:
        props_include = props

    # Loop through all labels and extract properties as np.ndarray
    for i, prop in enumerate(props_include):
        
        if i == 0: # First call
            
            features_i, names  = prop_to_allfeatures(
                prop,
                channels,
                params_channels['px_width'],
                brightness_channel = brightness_channel,
            )
            # Init features
            features = np.zeros((len(props), len(names)), dtype=np.float32)
            # Assignment to features
            features[i, :] = features_i

        else: # All others
            
            features_i = prop_to_allfeatures(
                prop,
                channels,
                params_channels['px_width'],
                brightness_channel = brightness_channel,
                )[0]
            # Assignment to features
            features[i, :] = features_i


    # Prepare dtype for conversion of features to structured numpy array
    dtypes = ['<f4' for name in names]
    # These fields will be stored as uint16 type
    names_uitype = ['id', 'x_mean', 'y_mean', 'x_min_bbox', 'x_max_bbox', 'y_min_bbox', 'y_max_bbox']
    names_uitype.extend(['x_max_' + name for name in channels.keys()])
    names_uitype.extend(['x_min_' + name for name in channels.keys()])

    # Change float32 to uint16 dtype as respectvie fields
    for i, name in enumerate(names):
        if name in names_uitype:
            dtypes[i] = '<u2'
    dtypes = [(name, dtype) for name, dtype in zip(names, dtypes)]

    # Convert features to structures dtype
    features = unstructured_to_structured(features, dtype=np.dtype(dtypes))


    return features


def crowns_add_features(
    channels: Dict,
    params_channels: Dict,
    crowns: Dict,
    params_crowns: Dict,
    params: Dict = {},
    ):

    # Define standard parameters
    params_standard = {
        'include_ids': None,
        'brightness_channel' : 'light',
        'exclude_chm_lower': 5,
        'exclude_chm_upper': 40,
    }
    # Assign standard parameters if not given
    for k in params_standard:
        if k in params:
            params[k] = params[k]
        else:
            params[k] = params_standard[k]

    # From polygons to labelimage
    labelimg = transformations.polygons_to_labelimage(crowns['polygons'], params_crowns['shape'])

    # Check if vegetation indices and hls images are in channels if not extend
    names_needed = ['ndvi', 'ndvire' ,'ndre', 'grvi', 'hue', 'light', 'sat']
    if not np.all([name in channels.keys() for name in names_needed]):
        transformations.channels_extend(channels)

    # Assing NaNs to all channels at pxs of low and high altitudes
    channels_cleanup = {}
    for key in channels:
        img = channels[key].copy()
        img[channels['chm'] < params['exclude_chm_lower']] = np.nan
        img[channels['chm'] > params['exclude_chm_upper']] = np.nan
        channels_cleanup[key] = img

    # Extract photometric features
    features = labelimage_extract_features(
        labelimg,
        channels_cleanup,
        params_channels,
        include_ids = params['include_ids'],
        brightness_channel  = params['brightness_channel'],
        )

    # Assign features to crowns
    crowns['features']['photometric'] = features

    # Assign parameters
    params_crowns['date_time_photometric'] = transformations.current_datetime()
    params_crowns['features_photometric_brightness_channel'] = params['brightness_channel']
    params_crowns['features_photometric_exclude_chm_lower'] = params['exclude_chm_lower']
    params_crowns['features_photometric_exclude_chm_upper'] = params['exclude_chm_upper']

    pass