""" Module to export channels, crowns as rois into .hdf5 files
"""
import os
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import h5py
import skimage.measure

import findatree.transformations as transformations


def pad_to_size(array, to_size:int=300):
    
    # Shape of orignal array
    shape = array.shape
    
    # Define one sided padding for first two dimensions
    pad_width = []
    for dim, size in enumerate(shape):
        
        assert size < to_size, f"Can not pad zeros, dimension {dim} is of size {size} > {to_size}"
        
        if dim < 2:
            pad_width.append([0, to_size-size])
        
        if dim == 2:
            pad_width.append([0, 0])
            
    return np.pad(array, pad_width=pad_width)


def export_rois(
    channels,
    params_channels,
    crowns,
    params_crowns,
    params_export,
    ):
    
    ########### Define
    
    # Assertions for export parameters
    assert 'channels_export' in params_export, f"Key 'channels_export' must be in params_export"
    assert 'query_export' in params_export, f"Key 'query_export' must be in params_export"
    assert 'size_export' in params_export, f"Key 'size_export' must be in params_export"
    
    # Extract original parameters 
    shape = params_channels['shape'] # to prepare multichannel image
    affine = params_channels['affine'] # to transform from image to geo coordinates
    affine = transformations.affine_numpy_to_resterio(affine)
    
    # Extract_parameters for export
    channels_export = params_export['channels_export'] # channel names
    n_channels_export = len(channels_export) # number of channels
    query_export = params_export['query_export']
    size_export = params_export['size_export']
    
    
    ########### Prepare
    
    # Prepare image
    image = np.zeros((shape[0], shape[1], n_channels_export), dtype=np.float32)
    for i, name in enumerate(channels_export):
        image[:,:,i] = channels[name]

    # Features as pd.DataFrame
    features = pd.merge(
        left = pd.DataFrame(crowns['features']['photometric']),
        right = pd.DataFrame(crowns['features']['terrestrial']),
        how= 'outer',
        on = ['id'],
        )

    # Apply query to features to get queryset ids
    ids = features.query(query_export).id.values
    
    assert len(ids) > 0, f"Query resulted in 0 polygons"

    # Reduce polygons to queryset ids
    polygons = {key: val for key, val in crowns['polygons'].items() if key in ids}

    # Transform the polygons to a regions labelimage
    labelimage = transformations.polygons_to_labelimage(polygons, shape=shape)
    
    
    ########### Extract
    
    # Get regionprops of labelimage
    props = skimage.measure.regionprops(labelimage, intensity_image=image)
    
    # Get ids, coordinate reference points, rois and masks
    ids = [prop.label for prop in props] # ids
    bboxs = [prop.bbox for prop in props]  # bounding boxes (image coordinates)
    coords = [[bbox[0], bbox[1]] for bbox in bboxs]  # coordinate reference points, upper left corner pixel (image coordinates)
    rois = [image[bbox[0]:bbox[2], bbox[1]:bbox[3], :] for bbox in bboxs]  # rois
    masks = [prop.image for prop in props]  # masks
    
    ########## Transform
    # Convert ids to np.ndarray
    ids = np.array(ids, dtype=np.uint16)
    
    # Convert rois to one large np.ndarray of fixed size
    rois_array = np.zeros((size_export, size_export, n_channels_export, len(rois)), dtype=np.float32)
    for i, roi in enumerate(rois):
        rois_array[:,:,:,i] = pad_to_size(roi, to_size=size_export)

    # Convert masks to one large np.ndarray of fixed size
    masks_array = np.zeros((size_export, size_export, len(masks)), dtype=np.float32)
    for i, mask in enumerate(masks):
        masks_array[:,:,i] = pad_to_size(mask, to_size=size_export)
    
    # rois with masked applied
    rois_masked_array = rois_array * masks_array[:,:,np.newaxis,:]
    
    # Convert coords from image to geo coordinates
    coords_geo = [affine * coord for coord in coords]
    
    # Convert coordinates to np.ndarrays
    coords = np.array(coords, dtype=np.uint16)
    coords_geo = np.array(coords_geo, dtype=np.float32)
    
    ########## Returns
    rois_dict = {
        'ids': ids,
        'coords': coords,
        'coords_geo': coords_geo,
        'images':rois_array,
        'masks': masks_array,
        'images_masked':rois_masked_array,
    }
    
    params_rois = {
        'channels': channels_export,
        'tnr': params_channels['tnr'],
        'affine': params_channels['affine'],
        'shape': params_channels['shape'],
        'number_rois': len(rois),
        'date_time':transformations.current_datetime(),
    }
    
        
    return rois_dict, params_rois


def rois_to_hdf5(
    rois,
    params_rois,
    dir_name: str='/home/flostehr/data/processed',
    ) -> None:

    # Define name of .hdf5
    name = f"tnr{params_rois['tnr']}_rois.hdf5"
    # Define full path to .hdf5, i.e. directory + name
    path = os.path.join(dir_name, name)
    # Define group name
    group_name = 'rois'

    # Now write group
    with h5py.File(path, 'w') as f:

        # Create main group
        grp = f.create_group(group_name)

        # Assign parameters as main group attributes
        for key in params_rois:
            grp.attrs[key] = params_rois[key]

        # Assign all channels as main group dsets
        for key in rois:
            grp.create_dataset(key, data=rois[key])

        pass

    
def load_rois_from_hdf5(
    path: str,
    load_sets: Union[List, None]=None,
    ) -> Tuple[Dict, Dict]:
    
    # Initialize data and parameters dictionary
    rois= {}
    params_rois = {}

    with h5py.File(path, 'r') as f:
        grp = f.get('rois')
        
        # If load_sets set to None load all sets
        if load_sets is None:
            load_sets = grp.keys()
            
        # Assign rois
        for key in grp.keys():
            if key in load_sets: 
                # Assign to rois
                rois[key] = grp.get(key)[()]
            
        # Assign params_rois
        for key in grp.attrs.keys():
            params_rois[key] = grp.attrs[key]
    
    return rois, params_rois

            
            
            
  