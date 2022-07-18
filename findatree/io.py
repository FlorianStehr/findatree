from typing import Dict, List, Tuple
import importlib
import os
import re
import h5py
import numpy as np
import shapefile
from tqdm import tnrange

import findatree.transformations as transformations
importlib.reload(transformations)

def find_all_tnrs_in_dir(dir_name):

    # Get all file names in dir
    file_names = [file for file in os.listdir(dir_name)]
    
    # Extract the five digit tnr number of all file names
    tnrs = [re.findall(r"tnr_\d{5}_", name, re.IGNORECASE)[0][4:-1] for name in file_names]

    return tnrs


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
        raise FileNotFoundError('Specific tnr number must be specified.')

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

        ########################################### Group
        # Get pointer to 'channels' group if exists
        if 'channels' in f:
            grp = f.get('channels')
            
            # Assert that any of the dsm, dtm and ortho paths saved in group attributes are the same as in params_channels
            for key in ['path_dsm', 'path_dtm', 'path_ortho']:
                assert grp.attrs[key] == params_channels[key], f"You are trying to overwrite channels with a different `{key}` attribute. No file was written. Change the saving directory."
        
        else:
            grp = f.create_group('channels')
        
        ########################################### Group attributes
        # Delete all old group attributes and ...
        for key in grp.attrs.keys():
            del grp.attrs[key]

        # ... assign (new) parameters as group attributes
        for key in params_channels:
            grp.attrs[key] = params_channels[key]

        ########################################### Group datasets
        # Delete all old group datasets and ...
        for key in grp.keys():
            del grp[key]

        # ... assign each channel as group dataset
        for key in channels:
            grp.create_dataset(key, data=channels[key])

        pass


#%%
def load_shapefile(dir_names: List, params_channels: Dict) -> Tuple[Dict, Dict]:

    # Search for paths to wanted files in given directories
    paths = _find_paths_in_dirs(dir_names, params_channels['tnr'])

    # Get affine geo-trafo as rasterio.Affine to convert geo to px coordinates
    affine = transformations.affine_numpy_to_resterio(params_channels['affine'])

    # Open shapefile
    with shapefile.Reader(paths['path_shapes']) as sf:

        # Read dimensions, names, etc.
        n_crowns = len(sf)
        attr_names = [f[0] for f in sf.fields[1:]]

        # Define which attributes will be included in final output
        attr_names_include = ['Enr', 'Bnr', 'Ba', 'BK','BHD_2020', 'Alter_2020', 'KKL', 'NBV', 'SST', 'Gilb', 'Kommentar', 'Sicherheit']
        

        # Init crowns polygons dictionary
        crowns_polys = {}

        # Init crowns records as numpy structed array
        dtype = transformations.geojson_records_fields_to_numpy_dtype(sf.fields, attr_names_include)
        crowns_recs = np.zeros(n_crowns, dtype = dtype)
        
        for idx, shape in enumerate(sf.shapes()):
            
            ##################### Polygons
            # Get shape's polygon
            poly = np.array(shape.points, dtype = np.float32).T

            # Convert polygon from geo to px coordinates
            poly = np.array(~affine * poly).T

            # Assign to crowns polygon dict
            crowns_polys[idx + 1] = poly

            ##################### Records
            # Get shape's records
            recs = sf.record(idx)

            # Reduce records to included attributes
            recs = [val for i, val in enumerate(recs) if attr_names[i] in attr_names_include]

            # Add id at beginning of records
            recs = [idx + 1] + recs

            # Assign to crowns records array
            crowns_recs[idx] = tuple(recs)


    # Create return dictionary of crowns_polys and crowns_recs
    crowns = {
        'polygons': crowns_polys,
        'features': dict([('terrestrial', crowns_recs)]),
    }
    
    # Create dictionary of parameters
    params = {}
    params['date_time_polygons'] = transformations.current_datetime()
    params['date_time_terrestrial'] = transformations.current_datetime()
    params['tnr'] = params_channels['tnr']
    params['affine'] = params_channels['affine']
    params['shape'] = params_channels['shape']
    params['path_shapes'] = paths['path_shapes']
    params['origin'] = 'human'
    params['number_crowns'] = n_crowns

    # Sort parameters according to key
    params = dict([(key, params[key]) for key in sorted(params.keys())])
    
    # Print parameters
    print('-----------')
    print('Parameters:')
    for k in params: print(f"  {k:<30}: {params[k]}")

    return crowns, params


#%%
def crowns_to_hdf5(crowns: Dict , params_crowns: Dict, dir_name: str=r'C:\Data\lwf\processed') -> None:
    
    # Nested function to savely get or create group
    def get_or_create_group(group_name, file):
        if group_name in file:
            grp = file.get(group_name)
        else:
            grp = file.create_group(group_name)
        return grp
    

    # Define name of .hdf5
    name = f"tnr{params_crowns['tnr']}.hdf5"
    
    # Define full path to .hdf5, i.e. directory + name
    path = os.path.join(dir_name, name)

    # Define group name
    try:
        group_name = 'crowns_' + params_crowns['origin'] 
    except:
        print("Please provide `'origin'` in params_crowns")


    # Open file for writing if exists, create otherwise.
    with h5py.File(path, 'a') as f:

        ######################################### Group
        # Get or create main group
        grp = get_or_create_group(group_name, f)
        

        ######################################### Group attributes
        # Delete all old group attributes and ...
        for key in grp.attrs.keys():
            del grp.attrs[key]

        # ... assign (new) parameters as group attributes 
        for key in params_crowns:
            grp.attrs[key] = params_crowns[key]
        
        ########################################### Polygons subgroup
        # Get or create 'polygons' subgroup
        grp = get_or_create_group(group_name + '/polygons', f)

        
        # Delete all old polygons subgroup datasets and ...
        for key in grp.keys():
            del grp[key]

        # ... assign polygons as datasets in 'polygons' subgroup
        for idx, poly in crowns['polygons'].items():  

            key = str(idx).zfill(5) # Convert idx to key to string with zero padding
            grp.create_dataset(key, data = poly)

        ########################################### Features subgroup
        # Get or create 'features' subgroup
        grp = get_or_create_group(group_name + '/features', f)

        # Assign all features (i.e. feature sets like 'terrestrial or 'photometric') in crowns['features'] as datasets in 'features' subgroup.
        for key, features in crowns['features'].items():
            
            # Assert that number of crowns in features are the same as number of crowns in params_crowns
            message = f"`len(crowns['features'][{key}]` is {features.shape[0]} but `params_crowns['number_crowns']` is {params_crowns['number_crowns']})"
            assert features.shape[0] == params_crowns['number_crowns'], message

            # Overwrite (if already exists) or write features as dataset
            if key in grp:
                del grp[key]
                dset = grp.create_dataset(key, data = features)    
            else:
                dset = grp.create_dataset(key, data = features)
            
            # Define attributes dict for features dataset
            features_attrs = {}
            features_attrs['names'] = [name for name in features.dtype.names]
            features_attrs['dtypes'] = [str(features.dtype[i]) for i in range(len(features.dtype))]

            # Write feature parameters into main group attributes
            grp = get_or_create_group(group_name, f)
            for key_attr, val_attr in features_attrs.items():
                grp.attrs['features_' + key + '_' + key_attr] = val_attr

                
        pass


#%% 
def load_hdf5(path: str, groups: List = ['channels', 'crowns_human', 'crowns_water'], features_only = False) -> Tuple[Dict, Dict]:

    # Initialize data and parameters dictionary
    data = {}
    data_params = {}

    with h5py.File(path, 'r') as f:
        
        for group in groups:

            ################################ Channels
            if group == 'channels':
                try: 
                    # Get `channels` group pointer
                    grp = f.get('channels')
                    # Assign `channels` datasets as dict to data
                    data['channels'] = dict([(key, grp.get(key)[()]) for key in grp.keys()])
                    # Assign `channels` attributes as dict to data_params
                    data_params['channels'] = dict([(key, grp.attrs[key]) for key in grp.attrs.keys()])
                except:
                    raise Warning(f"Group `{group}` not found.")

            ################################ Crowns
            if bool(re.search('crowns_*', group)):
                
                try:
                    # Get crowns group pointer
                    grp = f.get(group)
                    # Assign `crowns_*` attributes to params as dict
                    data_params[group] = dict([(key, grp.attrs[key]) for key in grp.attrs.keys()])
                    # Initialize crowns sub-dictionary
                    data_crowns = {}
                    
                    ################################ Features sub-group
                    # Get features sub-group pointer
                    grp = f.get(group + '/features')
                    # Get features as dict
                    features = dict([(key, grp.get(key)[()]) for key in grp.keys()])
                    # Convert features to numpy arrays with correct dtype
                    for key, val in features.items():
                        features[key] = np.array(val, dtype=val.dtype)
                    # Assign features to crowns sub-dictionary
                    data_crowns['features'] = features

                    ################################ Polygons sub-group
                    # Assign polygons datasets
                    if not features_only:
                        # Get polygons sub-group pointer
                        grp = f.get(group + '/polygons')
                        # Get features as dict
                        polygons = dict([(int(key), grp.get(key)[()]) for key in grp.keys()])
                        # Assign polygons to crowns sub-dictionary
                        data_crowns['polygons'] = polygons
                
                    
                    # Assign crowns sub-dictionary to data
                    data[group] = data_crowns

                except:
                    raise Warning(f"Group `{group}` not found.")
                
    return data, data_params
