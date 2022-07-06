from typing import Dict, List, Tuple
import importlib
import os
import re
import glob
import h5py
import numpy as np
import shapefile
import rasterio

import findatree.transformations as transformations
importlib.reload(transformations)


#%%
def _find_paths_in_dirs(
    dir_names: List[str],
    tnr_number: int=None,
    filetypes_pattern: str= '.(tif|shp)',
    dsm_pattern: str= 'dsm',
    dtm_pattern: str = 'dtm',
    ortho_pattern: str = 'ortho',
    shape_pattern: str = 'kr',
    ):
    """Find dsm, dtm and ortho rasters and corresponding shapefile (.shp) in given directories and return full unique paths.

    Parameters
    ----------
    dir_names : List[str]
        List of absolute paths to all folders containing the necessary dsm, dtm and ortho rasters.
    tnr_number : int, optional
        Area number to be loaded. If `None` all area numbers are shown, by default `None`.
    filetypes_pattern : str, optional
        Filetype pattern to be matched for rasters, by default '.(tif|shp)'.
    dsm_pattern : str, optional
        dsm file pattern to be matched for rasters, by default 'dsm'.
    dtm_pattern : str, optional
        dtm file pattern to be matched for rasters, by default 'dtm'.
    shape_pattern : str, optional
        shape files pattern to be matched for delineated crown shape files, by default 'kr'.
    verbose : bool, optional
        Print parameters during call, by default True.

    Returns
    -------
    paths_dict: Dict
        * tnr [int]: Area number
        * path_dsm [str]: Absolute path to dsm raster
        * path_dtm [str]: Absolute path to dtm raster
        * path_ortho [str]: Absolute path to ortho raster
        * paths_shapes [dict]: Absolute paths to delineated crown shape files with keys ['dbf','shp','shx'].
    """

    # Get sorted list of full paths to all files in all directories
    paths = []
    for dir_name in dir_names:
            path = [os.path.join(dir_name, file) for file in os.listdir(dir_name)]
            paths.extend(path)
    
    # Sort paths for viewing
    paths = sorted(paths)

    # Print all paths if `tnr_number` was None
    if tnr_number is None:
        print('No tnr found, displaying all available files:')
        for p in paths: print('  ' + p)
        print()

    # Define tnr_pattern
    tnr_number = str(tnr_number)
    tnr_pattern = 'tnr_' + tnr_number

    # Reduce to paths that match `tnr_pattern`
    paths = [p for p in paths if bool(re.search(tnr_pattern, os.path.split(p)[-1], re.IGNORECASE))]

    # Reduce to paths that match `filetypes_pattern`
    paths = [p for p in paths if bool(re.search(filetypes_pattern, os.path.splitext(p)[-1], re.IGNORECASE))]
    
    # Get path that contains non-case sensitive `dsm_pattern`
    paths_dsm = [p for p in paths if bool(re.search(dsm_pattern, os.path.split(p)[-1], re.IGNORECASE))]

    # Get path that contains non-case sensitive `dtm_pattern`
    paths_dtm = [p for p in paths if bool(re.search(dtm_pattern, os.path.split(p)[-1], re.IGNORECASE))]

    # Get path that contains non-case sensitive `ortho_pattern`
    paths_ortho = [p for p in paths if bool(re.search(ortho_pattern, os.path.split(p)[-1], re.IGNORECASE))]
    
    # Get path that contains non-case sensitive `shape_pattern` (.shp file only, see filtype_pattern)
    paths_shape = [p for p in paths if bool(re.search(shape_pattern, os.path.split(p)[-1], re.IGNORECASE))]

    # Assert that there is only one valid dsm, dtm and ortho file 
    assert len(paths_dsm) == 1, f"No file or more than one file with pattern `{dsm_pattern}` found in given directories"
    assert len(paths_dtm) == 1, f"No file or more than one file with pattern `{dtm_pattern}` found in given directories"
    assert len(paths_ortho) == 1, f"No file or more than one file with pattern `{ortho_pattern}` found in given directories"
    assert len(paths_shape) == 1, f"No file or more than one file with pattern `{shape_pattern}*.shp` found in given directories"

    # Join tnr number and dsm/dtm/ortho paths
    paths_dict = {
        'tnr': tnr_number,
        'path_dsm': paths_dsm[0],
        'path_dtm': paths_dtm[0],
        'path_ortho': paths_ortho[0],
        'path_shapes': paths_shape[0],
    }

    return paths_dict


#%%
def channels_to_hdf5(channels: Dict , params_channels: Dict, dir_name: str=r'C:\Data\lwf\processed') -> None:
    """Save all secondary channels in .hdf5 container.

    Group `channels` wil be created in .hdf5 with `channels` as datasets and `params_channels` as group attributes.
    If .hdf5 already exists and dsm, dtm and ortho paths did not change the file is overwritten, otherwise an error is raised.

    Parameters
    ----------
    channels : Dict
        Dict. of all normalized secondary channels as returned by findatree.geo_to_image.channels_primary_to_secondary()).
    params_channels : Dict
        Dict. of parameters as returned by geo_to_image.findatree.load_channels().
    dir_name: str
        Path to directory where .hdf5 is stored.
    """
    # Define name of .hdf5
    name = f"tnr{params_channels['tnr']}.hdf5"
    
    # Define full path to .hdf5, i.e. directory + name
    path = os.path.join(dir_name, name)

    # Open file for writing if exists, create otherwise.
    with h5py.File(path, 'a') as f:
        '''Three possible cases to cover:
            1. Channels group EXISTS ...
                a. ... and dsm, dtm and ortho path attributes are NOT the same as in params_channels -> Raise Error
                b. ... and dsm, dtm and ortho path attributes ARE the same as in params_channels -> Continue and overwrite 
            2. Channels group does NOT EXIST -> Write
        '''
        if 'channels' in f:
            grp = f.get('channels')
            
            # Assert that any of the dsm, dtm and ortho paths saved in group attributes are the same as in params_channels
            for key in ['path_dsm', 'path_dtm', 'path_ortho']:
                assert grp.attrs[key] == params_channels[key], f"You are trying to overwrite channels with a different `{key}` attribute. No file was written. Change the saving directory."
        
        else:
            grp = f.create_group('channels')
        
        # Update all parameters as group attributes
        for key in params_channels:
            grp.attrs[key] = params_channels[key]

        # Add/update all channel as group datasets
        for key in channels:
            
            # Overwrite case [1b]
            if key in grp: 
                del grp[key]
                grp.create_dataset(key, data=channels[key])
            
            # Write case [2]
            else:
                dset = grp.create_dataset(key, data=channels[key])
        pass


#%%
def load_shapefile(dir_names: List, params_channels: Dict) -> Tuple[Dict, Dict, Dict]:

    # Search for paths to wanted files in given directories
    paths = _find_paths_in_dirs(dir_names, params_channels['tnr'])

    # Get affine geo-trafo as rasterio.Affine to convert geo to px coordinates
    affine = transformations.affine_numpy_to_resterio(params_channels['affine'])

    # Open shapefile
    with shapefile.Reader(paths['path_shapes']) as sf:

        # Read dimensions, names, etc.
        n_shapes = len(sf)
        attr_names = [f[0] for f in sf.fields[1:]]
        n_attr = len(sf.record(0))

        # Define which attributes will be included in final output
        attr_names_include = ['Enr', 'Bnr', 'Ba', 'BHD_2020', 'Alter_2020', 'BK', 'KKL', 'NBV', 'SST', 'Gilb', 'Kommentar', 'Sicherheit']
     
        # Assert that attribute names match length of records
        assert len(attr_names) == n_attr, f"`len(shape_attr_names)` of {len(attr_names)} does not match `n_shape_attr` of {n_attr}"

        # Now go through all shapes
        crowns = {}
        for idx, shape in enumerate(sf.shapes()):
            
            ##################### Polygons
            # Get shape's polygon
            poly = np.array(shape.points).T

            # Convert polygon from geo to px coordinates
            poly = np.array(~affine * poly).T

            # Prepare polygon dict
            poly_dict = dict([('polygon', poly)])

            ##################### Records
            # Get shape's records
            records = sf.record(idx)

            # Create records dictionary, i.e. attributes
            attrs = [ (attr_names[i].lower(), val) for i, val in enumerate(records) if attr_names[i] in attr_names_include ]
            attrs_dict = dict([('attributes', dict(attrs))])


            # Add merged polygon/attributes dicts to shapes dict
            crowns[idx + 1] = poly_dict | attrs_dict  # Reserve 0 for nodata values
    
    # Create dictionary of parameters
    params = {}
    params['tnr'] = params_channels['tnr']
    params['affine'] = params_channels['affine']
    params['path_shapes'] = paths['path_shapes']
    params['origin'] = 'human'
    params['number_shapes'] = n_shapes
    params['number_attributes'] = len(attr_names_include)
    params['attribute_names'] = [name.lower() for name in attr_names_include]

    return crowns, params



#%%
def segments_to_hdf5(segments: Dict , params_segments: Dict, dir_name: str=r'C:\Data\lwf\processed') -> None:
    """Save all segmentation maps in .hdf5 container.

    Group `segments` wil be created in .hdf5 with `channels` as datasets and `params_segments` as group attributes.

    Parameters
    ----------
    segments : Dict
        Dictionary of labels as returned by segmentation.segment()
    params_segments : Dict
        Dictionary ofsegmentation parameters as returned by segmentation.segment()
    dir_name: str
        Path to directory where .hdf5 is stored
    """
    # Define name of .hdf5
    name = f"tnr{params_segments['tnr']}.hdf5"
    
    # Define full path to .hdf5, i.e. directory + name
    path = os.path.join(dir_name, name)

    # Open file for writing if exists, create otherwise.
    with h5py.File(path, 'a') as f:
        '''Two possible cases to cover:
            1. Segments group EXISTS -> Continue and overwrite 
            2. Segments group does NOT EXIST -> Write
        '''
        if 'segments' in f:
            grp = f.get('segments')
        else:
            grp = f.create_group('segments')
        
        # Update all parameters as group attributes
        for key in params_segments:
            grp.attrs[key] = params_segments[key]

        # Add/update all channel as group datasets
        for key in segments:

            # Overwrite case [1]
            if key in grp: 
                del grp[key]
                grp.create_dataset(key, data=segments[key])
            
            # Write case [2]
            else:
                dset = grp.create_dataset(key, data=segments[key])
        pass