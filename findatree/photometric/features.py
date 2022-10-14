from typing import List, Tuple, Dict
import numpy as np
import warnings
import importlib
import skimage.measure
from numpy.lib.recfunctions import unstructured_to_structured
from tqdm import tqdm

import findatree.transformations as transformations
import findatree.photometric.shadow as shadow

importlib.reload(transformations)
importlib.reload(shadow)

#%%
def prop_to_distances(prop):
    distance_names = [
        'perimeter',
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
    

def prop_to_idxs(prop, channels):

    # Get indices of labeled region
    idxs = prop['coords']
    idxs = (idxs[:,0], idxs[:,1])
    
    # Blue values of the labeled region (shadow mask was already applied in all channels!)
    vals = channels['blue'][idxs]
    
    # Non-shadow, i.e. bright pixel indices
    idxs_bright = (
        idxs[0][np.isfinite(vals)],
        idxs[1][np.isfinite(vals)],
    )
    
    return idxs, idxs_bright
    

#%%
def idxs_to_areas(idxs, idxs_bright):
    
    ##### Area of all pixels in the labeled region
    area = len(idxs[0])    
    
    ##### Area of the non-shadow pixels
    # Area of non-shadow pixels (i.e. finite values) in the labeled region
    area_bright = len(idxs_bright[0])

    # Add values and names to output
    areas = np.array([area, area_bright], dtype=np.float32)
    area_names = ['area', 'area_bright']
    
    return areas, area_names


def idxs_to_coords(idxs, idxs_bright, channels):
    
    coord_names_max = ['chm', 'avg']
    coord_names = []
    coords = []
    
    #### All pixels
    # Add (unweighted) center coordinate x in pixels (mean)
    coord_names.append('x_mean')
    coords.append(np.mean(idxs[1]))

    # Add (unweighted) center coordinate y in pixels (mean)
    coord_names.append('y_mean')
    coords.append(np.mean(idxs[0]))

    # Add bounding box minimum of all pixels in x in pixels
    coord_names.append('x_min_bbox')
    coords.append(np.min(idxs[1]))

    # Add bounding box maximum of all pixels in x in pixels
    coord_names.append('x_max_bbox')
    coords.append(np.max(idxs[1]))

    # Add bounding box minimum of all pixels in y in pixels
    coord_names.append('y_min_bbox')
    coords.append(np.min(idxs[0]))

    # Add bounding box maximum of all pixels in y in pixels
    coord_names.append('y_max_bbox')
    coords.append(np.max(idxs[0]))
    
    #### Coordinates of maximum intensity
    # Add coordinate x of pixel with maximum intensity
    coord_names.extend(['x_max_' + name for name in coord_names_max])
    try:
        coords.extend([idxs[1][np.nanargmax(channels[name][idxs])] for name in coord_names_max])
    except:
        coords.extend([np.nan for name in coord_names_max])

    # Add coordinate y of pixel with maximum intensity
    coord_names.extend(['y_max_' + name for name in coord_names_max])
    try:
        coords.extend([idxs[0][np.nanargmax(channels[name][idxs])] for name in coord_names_max])
    except:
        coords.extend([np.nan for name in coord_names_max])
        
    
    #### Non shadow pixels
    if len(idxs_bright[0]) > 0:
        # Add bounding box minimum of all pixels in x in pixels
        coord_names.append('x_min_bbox_bright')
        coords.append(np.min(idxs_bright[1]))

        # Add bounding box maximum of all pixels in x in pixels
        coord_names.append('x_max_bbox_bright')
        coords.append(np.max(idxs_bright[1]))

        # Add bounding box minimum of all pixels in y in pixels
        coord_names.append('y_min_bbox_bright')
        coords.append(np.min(idxs_bright[0]))

        # Add bounding box maximum of all pixels in y in pixels
        coord_names.append('y_max_bbox_bright')
        coords.append(np.max(idxs_bright[0]))
    else:
        coord_names.extend(['x_min_bbox_bright', 'x_max_bbox_bright', 'y_min_bbox_bright', 'y_max_bbox_bright'])
        coords.extend([np.nan]*4)
        
    return coords, coord_names
    
'''  
#%%
def prop_to_intensitycoords(prop, channels):
    
    ####################### Define image-indices and channel values of the segment

    # Channel names that are used for computation of:
    #    * intensity weighted metrics
    #    * xy_max, i.e. location of pixel with maximum intensity
    names_intensity = [
        'chm',                              # canopy height model
        'light', 'sat', 'hue',              # hls color space: lightness, saturation, hue
        'ndvi', 'ndvire', 'ndre', 'grvi',   # veg. indices
        'blue','green','red','re','nir',    # absolute colors
        'gob','rob','reob','nob',           # color ratios normalized to blue
        'avg',                              # average over all channels
        ]

    names_xy_max = ['chm', 'light']
    
    # These are all the channels for preparation of the flat channel array
    names = names_xy_max.copy()
    names.extend([name for name in names_intensity if name not in names_xy_max])

    # Indices of the label, in image coordinates
    idxs = prop['coords']
    idxs = (idxs[:,0], idxs[:,1])

    # Prepare flat array of values of all channels specified by names. Row corresponds to channel.
    channels_flat = np.zeros(
        (len(names), len(idxs[0])),
        dtype=np.float32,
        )
    for i, name in enumerate(names):
        img = channels[name]
        channels_flat[i,:] = img[idxs]

    ####################### Define brightest indices and values of the segment

    # Which row in channels_flat corresponds to brightness channel?
    row_bright = names.index(bright_channel)
    
    # Get brightness values
    bright_vals = channels_flat[row_bright, :]

    # Set the threshold for bright pixels to upper 75 percentile of brightness values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        bright_thresh = np.nanpercentile(bright_vals, 75)

    # Get columns where brightness values are above threshold, in channels_flat coordinates.
    cols_bright = np.where(bright_vals > bright_thresh)[0]
    
    # Treat case of zero brightest pixels -> Assign index of maximum value
    if len(cols_bright) == 0:
        cols_bright = [np.argmax(bright_vals)]
    
    # These are the indices of the brightest pixels, in image coordinates.
    idxs_bright = (idxs[0][cols_bright], idxs[1][cols_bright])

    # These are the brightest values of all channels_flat
    channels_flat_bright = channels_flat[:, cols_bright]


    ####################### Feature extraction

    # Init data and names
    features = []
    names_features = []

    # This is used to catch all the numpy warnings caused by numpy.nanmean() and alike 
    # when all values in one crown are NaNs. Can be ignored, because NaNs are assigned that can be filtered out later.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        ############# All pixels in segment

        # Add minimum value of each channel
        names_features.extend(['min_' + name for name in names_intensity])
        features.extend( list( np.nanmin(channels_flat, axis=1) ) )

        # Add maximum value of each channel
        names_features.extend(['max_' + name for name in names_intensity])
        features.extend( list( np.nanmax(channels_flat, axis=1) ) )

        # Add mean value of each channel
        names_features.extend(['mean_' + name for name in names_intensity])
        features.extend( list( np.nanmean(channels_flat, axis=1) ) )

        # Add std value of each channel
        names_features.extend(['std_' + name for name in names_intensity])
        features.extend( list( np.nanstd(channels_flat, axis=1) ) )

        # Add median value of each channel
        names_features.extend(['median_' + name for name in names_intensity])
        features.extend( list( np.nanmedian(channels_flat, axis=1) ) )

        # Add 75 percentile value of each channel
        names_features.extend(['perc75_' + name for name in names_intensity])
        features.extend( list( np.nanpercentile(channels_flat, 75, axis=1) ) )

        # Add 25 percentile value of each channel
        names_features.extend(['perc25_' + name for name in names_intensity])
        features.extend( list( np.nanpercentile(channels_flat, 25, axis=1) )



        ############# Brightest pixels in segment

        # Add minimum value of each channel of brightest pixels
        names_features.extend(['min_bright_' + name for name in names_intensity])
        features.extend( list( np.nanmin(channels_flat_bright, axis=1) ) )

        # Add maximum value of each channel of brightest pixels
        names_features.extend(['max_bright_' + name for name in names_intensity])
        features.extend( list( np.nanmin(channels_flat_bright, axis=1) ) )

        # Add mean value of brightest pixels of each channel 
        names_features.extend(['mean_bright_' + name for name in names_intensity])
        features.extend( list( np.nanmean(channels_flat_bright, axis=1) ) )

        # Add std value of brightest pixels of each channel
        names_features.extend(['std_bright_' + name for name in names_intensity])
        features.extend( list( np.nanstd(channels_flat_bright, axis=1) ) )

        # Add median value of brightest pixels of each channel
        names_features.extend(['median_bright_' + name for name in names_intensity])
        features.extend( list( np.nanmedian(channels_flat_bright, axis=1) ) )

        # Add 75 percentile value of brightest pixels of each channel
        names_features.extend(['perc75_bright_' + name for name in names_intensity])
        features.extend( list( np.nanpercentile(channels_flat_bright, 75, axis=1) ) )

        # Add 25 percentile value of brightest pixels of each channel
        names_features.extend(['perc25_bright_' + name for name in names_intensity])
        features.extend( list( np.nanpercentile(channels_flat_bright, 25, axis=1) ) )


    ############# Create final output array

    features = np.array(features, dtype=np.float32)

    return features, names_features

'''

#%%
def prop_to_allfeatures(prop, channels, px_width):
    
    # Get label
    label_name = ['id']                 # We will store the crown identifier as `id` ...
    label = np.array([ prop['label'] ]) # ... but in skimage it's called `label`
    
    # Get distances
    distances, distance_names = prop_to_distances(prop)
    distances = distances * px_width # Unit conversion from px to [m]
    
    # Get ratios
    ratios, ratio_names = prop_to_ratios(prop)
    
    # Get indices of all and non-shadow pixels in labeled region
    idxs, idxs_bright = prop_to_idxs(prop, channels)
    
    # Get areas
    areas, area_names = idxs_to_areas(idxs, idxs_bright)
    areas = areas * px_width**2 # Unit conversion from px to [m**2]
    
    # Get coordinates of all and non-shadow pixels in labeled region
    coords, coord_names = idxs_to_coords(idxs, idxs_bright, channels)

    # Get intensities and coordinates
    # intcoords, intcoord_names = prop_to_intensitycoords(
        # prop,
        # channels, 
        # )

    # Concatenate all props and corresponding names
    features = np.concatenate(
        (label,
        distances,
        ratios,
        areas,
        coords,
        # intcoords,
        ),
    )
    names = label_name + distance_names + ratio_names + area_names + coord_names #+ intcoord_names

    return features, names

#%%

def labelimage_extract_features(
    labelimg,
    channels,
    params_channels,
    include_ids = None,
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
                )[0]
            # Assignment to features
            features[i, :] = features_i

    '''
    # Assign decays
    decay_channel_names = ['chm', 'avg']

    for channel_name in decay_channel_names:
        # Get peak coordinates
        row_peak_idx = features[:, names.index('y_max_' + channel_name)].astype(np.uint16)
        col_peak_idx = features[:, names.index('x_max_' + channel_name)].astype(np.uint16)  

        # Loop through all decay channels and collect decay at radius from peak
        for i in [int(2**i) for i in range(5)]:
            # Name of decay_channel and append to names
            decay_name = channel_name + f"_decay{i}"
            names.append(decay_name)

            # Get decay values at peak and add to features
            peak_decay = channels[decay_name][row_peak_idx, col_peak_idx].reshape(-1,1)
            features = np.concatenate([features, peak_decay], axis=1)
    '''

    # Prepare dtype for conversion of features to structured numpy array
    dtypes = ['<f4' for name in names]

    # These fields will be stored as uint16 type
    names_uitype = [
        'id',
        'x_mean', 'y_mean',
        'x_min_bbox', 'x_max_bbox', 'y_min_bbox', 'y_max_bbox',
        'x_min_bbox_bright', 'x_max_bbox_bright', 'y_min_bbox_bright', 'y_max_bbox_bright',
        ]
    names_uitype.extend(['x_max_' + name for name in channels.keys()])
    names_uitype.extend(['x_min_' + name for name in channels.keys()])
    names_uitype.extend(['y_max_' + name for name in channels.keys()])
    names_uitype.extend(['y_min_' + name for name in channels.keys()])

    # Change float32 to uint16 dtype as respective fields
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
        'shadowmask_channel' : 'avg',
        'shadowmask_width': 101,
        'shadowmask_thresh_chm_lower': 5,
        'shadowmask_thresh_chm_upper': 40,
    }
    # Assign standard parameters if not given
    for k in params_standard:
        if k in params:
            params[k] = params[k]
        else:
            params[k] = params_standard[k]

    # From polygons to labelimage
    labelimg = transformations.polygons_to_labelimage(crowns['polygons'], params_crowns['shape'])

    # Define needed channels
    names_needed = [
        'chm',                              # canopy height model
        'light', 'sat', 'hue',              # hls color space: lightness, saturation, hue
        'ndvi', 'ndvire', 'ndre', 'grvi',   # veg. indices
        'gob','rob','reob','nob',           # color ratios normalized to blue
        'avg',                              # Average over all color channels
        ]
    
    # If any of the needed channels is non-existent extend
    if not np.all([name in channels.keys() for name in names_needed]):
        transformations.channels_extend(channels)
    
    # Compute shadow mask based on local otsu thresholding
    mask = shadow._mask_local_otsu(
        channels,
        params['shadowmask_channel'],
        params['shadowmask_width'],
        params['shadowmask_thresh_chm_lower'],
        params['shadowmask_thresh_chm_upper'],
    )
    
    # Assing NaNs to to shadow pixels in all channels
    channels_cleanup = {}
    for key in channels:
        img = channels[key].copy()
        img[~mask] = np.nan
        channels_cleanup[key] = img

    # Extract photometric features
    features = labelimage_extract_features(
        labelimg,
        channels_cleanup,
        params_channels,
        include_ids = params['include_ids'],
        )

    # Assign features to crowns
    crowns['features']['photometric'] = features

    # Assign parameters
    params_crowns['date_time_photometric'] = transformations.current_datetime()
    params_crowns['features_photometric_shadowmask_channel'] = params['shadowmask_channel']
    params_crowns['features_photometric_shadowmask_width'] = params['shadowmask_width']
    params_crowns['features_photometric_shadowmask_thresh_chm_lower'] = params['shadowmask_thresh_chm_lower']
    params_crowns['features_photometric_shadowmask_thresh_chm_upper'] = params['shadowmask_thresh_chm_upper']

    pass