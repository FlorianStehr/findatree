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
    
    # Non-shadow, i.e. bright pixel indices <-> finite values after assigment of NaNs to shadow pixels
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
    
    coord_names_max = ['chm']
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
    coord_names.extend(['x_min_bbox_bright', 'x_max_bbox_bright', 'y_min_bbox_bright', 'y_max_bbox_bright'])
    
    if len(idxs_bright[0]) > 0:
        
        # Add bounding box minimum of all pixels in x in pixels
        coords.append(np.min(idxs_bright[1]))

        # Add bounding box maximum of all pixels in x in pixels
        coords.append(np.max(idxs_bright[1]))

        # Add bounding box minimum of all pixels in y in pixels
        coords.append(np.min(idxs_bright[0]))

        # Add bounding box maximum of all pixels in y in pixels
        coords.append(np.max(idxs_bright[0]))
    
    else:
        coords.extend([np.nan]*4)
    
    # Convert list of coordinates to np.ndarray 
    coords = np.array(coords, dtype=np.float32)
    
    
    return coords, coord_names
    

#%%
def idxs_to_intensities(idxs, channels):
   
    # Channel names that are used for computation of intensity metrics
    names = [
        'chm',                                      # canopy height model
        'blue','green','red','re','nir',            # absolute colors
        'avg',                                      # average of all absolute colors
        'nblue','ngreen','nred','nre',              # color ratios normalized to NIR
        'light', 'sat', 'hue',                      # hls color space: lightness, saturation, hue
        'ndvi', 'ndvire', 'ndre', 'grvi', 'evi',    # veg. indices
        ]
        
    # Percentile values that are calculated for each intensity
    percs = [5, 25, 50, 75, 95]
    
    # Total number of features that are computed
    n_features = len(names) * len(percs)
    
    
    # Assign intensities and names
    
    # Init data and names
    intensities = []
    intensity_names = []
    
    # Extend intensity_names
    intensity_names.extend(
        [f"perc{perc}_" + name for name in names for perc in percs]
    )
    
    # Extend intensities treating case of completely shadowed regions
    if len(idxs[0]) > 0:
        intensities.extend(
            [np.percentile(channels[name][idxs], perc) for name in names for perc in percs]
        )
        
    else:
        intensities.extend([np.nan]*n_features)

    # Convert list of intensities to np.ndarray 
    intensities = np.array(intensities, dtype=np.float32)

    return intensities, intensity_names


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

    # Get intensities of non-shadow pixels only
    intensities, intensity_names = idxs_to_intensities(
        idxs_bright,
        channels, 
        )

    # Concatenate all props and corresponding names
    features = np.concatenate(
        (label,
        distances,
        ratios,
        areas,
        coords,
        intensities,
        ),
    )
    names = label_name + distance_names + ratio_names + area_names + coord_names + intensity_names

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
    
    #### Compute features based on regionprops
    
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

    
    #### Add height decay features

    # Get height peak coordinates
    row_peak_idx = features[:, names.index('y_max_chm')].astype(np.uint16)
    col_peak_idx = features[:, names.index('x_max_chm')].astype(np.uint16)  

    # Loop through all decay channels and collect decay at radius from peak
    for i in [int(2**i) for i in range(5)]:
        
        # Define decay name and add to feature names
        decay_name = f"decay{i}"
        names.append(decay_name)

        # Get decay values at peak and add to features
        peak_decay = channels[decay_name][row_peak_idx, col_peak_idx].reshape(-1,1)
        features = np.concatenate([features, peak_decay], axis=1)
            
    
    #### dtype conversions
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
        'chm',                                              # canopy height model
        'blue','green','red','re','nir',                    # absolute colors
        'avg',                                              # average of all absolute colors
        'nblue','ngreen','nred','nre',                      # color ratios normalized to NIR
        'light', 'sat', 'hue',                              # hls color space: lightness, saturation, hue
        'ndvi', 'ndvire', 'ndre', 'grvi', 'evi',            # veg. indices
        'decay1', 'decay2', 'decay4', 'decay8', 'decay16',  # height decays around maximum pixel
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
    
    # Assign mask to channels 
    channels['mask'] = mask.astype(np.float32)
     
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