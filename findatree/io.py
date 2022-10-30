from typing import Dict, List, Tuple
import importlib
import os
import yaml
import re
import h5py
import numpy as np
import pandas as pd
import shapefile
from tqdm import tnrange

import findatree.transformations as transformations
importlib.reload(transformations)


#%%
def list_of_dicts_to_yaml(path: str, list_of_dicts: List[Dict]):
    with open(path, "w") as f:
        yaml.dump_all(list_of_dicts, f, default_flow_style=False)
    
    pass


#%%
def find_all_tnrs_in_dir(
    dir_name,
    tnr_pattern_lead: str = 'tnr_',
    tnr_pattern_trail: str = '_',
    ):

    # Get all file names in dir
    file_names = [file for file in os.listdir(dir_name)]
    
    # Extract the two to five digit tnr number of all file names
    tnrs = [re.findall(tnr_pattern_lead + r"\d{2}" + tnr_pattern_trail, name, re.IGNORECASE) for name in file_names]
    tnrs.extend( [re.findall(tnr_pattern_lead + r"\d{3}" + tnr_pattern_trail, name, re.IGNORECASE) for name in file_names] )
    tnrs.extend( [re.findall(tnr_pattern_lead + r"\d{4}" + tnr_pattern_trail, name, re.IGNORECASE) for name in file_names] )
    tnrs.extend( [re.findall(tnr_pattern_lead + r"\d{5}" + tnr_pattern_trail, name, re.IGNORECASE) for name in file_names] )

    # Remove zero length entries
    tnrs = [tnr[0] for tnr in tnrs if len(tnr) > 0]

    # Remove the leading 'tnr_'
    tnrs = [tnr[len(tnr_pattern_lead):] for tnr in tnrs]

    # Remove the trailing '_'
    tnrs = [tnr[:-len(tnr_pattern_trail)] for tnr in tnrs]

    # Convert tnrs from strings to integers
    tnrs = [int(tnr) for tnr in tnrs]

    # Get unique tnr numbers
    tnrs = list(np.unique(tnrs))

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
    tnr_pattern = 'tnr_' + tnr_number + '_'

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
    assert len(paths_dsm) == 1, f"(tnr{tnr_number}) No file or more than one file with pattern `{dsm_pattern}` found in given directories"
    assert len(paths_dtm) == 1, f"(tnr{tnr_number}) No file or more than one file with pattern `{dtm_pattern}` found in given directories"
    assert len(paths_ortho) == 1, f"(tnr{tnr_number}) No file or more than one file with pattern `{ortho_pattern}` found in given directories"

    # Join tnr number and dsm/dtm/ortho paths
    paths_dict = {
        'tnr': tnr_number,
        'path_dsm': paths_dsm[0],
        'path_dtm': paths_dtm[0],
        'path_ortho': paths_ortho[0],
    }
    try:
        paths_dict['path_shapes'] = paths_shape[0]
    except:
        pass

    return paths_dict


#%%
def _hdf5_yield_dset_names(file, object):
    
    if isinstance(object, h5py.Group):
        for sub_object in object.values():
            if isinstance(sub_object, h5py.Dataset):
                yield sub_object.name
            else:
                yield from _hdf5_yield_dset_names(file, sub_object)

#%%
def _hdf5_yield_subgroup_names(file, object):
    
    if isinstance(object, h5py.Group):
        for sub_object in object.values():
            if isinstance(sub_object, h5py.Group):
                yield sub_object.name    
            else:
                yield from _hdf5_yield_subgroup_names(file, sub_object)


#%%
def hdf5_names_tolist(path, group_name):

    with h5py.File(path, 'a') as f:
        
        if group_name in f.keys():

            # Get the full paths to all dsets in all subgroups
            dset_names_generator = _hdf5_yield_dset_names(f, f[group_name])
            dset_names = [name for name in dset_names_generator]

            # Get the full paths to all subgroups in all subgroups
            subgroup_names_generator = _hdf5_yield_subgroup_names(f, f[group_name])
            subgroup_names = [f"/{group_name}"] # Add group itself to subgroup_names
            subgroup_names.extend([name for name in subgroup_names_generator])

            # Reduce to unique subgroups
            subgroup_names = list(np.unique(subgroup_names))
            
            # Get the depth of each subgroup
            subgroup_depth = [name.count('/') for name in subgroup_names]

            # Sort subgroups according to depth (high to low)
            subgroup_depth_sortidx = list(np.argsort(subgroup_depth))
            subgroup_depth_sortidx.reverse()
            subgroup_names = [subgroup_names[idx] for idx in subgroup_depth_sortidx]

        else:
            dset_names = []
            subgroup_names = []

    return dset_names, subgroup_names

#%%
def hdf5_delete_group(path, group_name):

    with h5py.File(path, 'a') as f:
        
        dset_names, subgroup_names = hdf5_names_tolist(path, group_name)

        # Now first delete all dsets
        # if len(dset_names) > 0:
        for dset_name in dset_names:
            del f[dset_name]

        # ... then delete all subgroups.
        # if len(subgroup_names) > 0:
        for subgroup_name in subgroup_names: 
            del f[subgroup_name]

    pass

#%%
def channels_to_hdf5(channels: Dict , params_channels: Dict, dir_name: str=r'C:\Data\lwf\processed') -> None:
    """Save all secondary channels in .hdf5 container.

    Group `channels` wil be created in .hdf5 with `channels` as datasets and `params_channels` as group attributes.

    Parameters
    ----------
    channels : Dict
        Dict. of all normalized secondary channels as returned by findatree.geo_to_image.channels_primary_to_secondary()).
    params_channels : Dict
        Dict. of parameters as returned by geo_to_image.findatree.load_channels().
    dir_name: str
        Path to directory where .hdf5 is stored.
    """

    # Remove all non-basic channels like 'ndvi', 'light', etc.
    channels_save = {}
    for key in channels:
        if key in ['blue', 'green', 'red', 're', 'nir', 'chm']:
            channels_save[key] = channels[key].copy()

    # Define name of .hdf5
    name = f"tnr{params_channels['tnr']}.hdf5"
    # Define full path to .hdf5, i.e. directory + name
    path = os.path.join(dir_name, name)
    # Define group name
    group_name = 'channels'

    # Completely delete group and all of it's subgroups, dsets and respective attributes
    hdf5_delete_group(path, group_name)

    # Now write group
    with h5py.File(path, 'a') as f:

        # Create main group
        grp = f.create_group(group_name)

        # Assign parameters as main group attributes
        for key in params_channels:
            grp.attrs[key] = params_channels[key]

        # Assign all channels as main group dsets
        for key in channels_save:
            grp.create_dataset(key, data=channels_save[key])

        pass


#%%
def crowns_to_hdf5(crowns: Dict , params_crowns: Dict, dir_name: str=r'C:\Data\lwf\processed') -> None:
    
    # Define name of .hdf5
    name = f"tnr{params_crowns['tnr']}.hdf5"
    # Define full path to .hdf5, i.e. directory + name
    path = os.path.join(dir_name, name)
    # Define group name
    try:
        group_name = 'crowns_' + params_crowns['origin'] 
    except:
        print("Please provide `'origin'` in params_crowns")

    # Completely delete group and all of it's subgroups, dsets and respective attributes
    hdf5_delete_group(path, group_name)

    # Open file for writing if exists, create otherwise.
    with h5py.File(path, 'a') as f:

        ######################################### Group
        # Create main group
        grp_main = f.create_group(group_name)

        # Assign parameters as main group attributes
        for key in params_crowns:
            grp_main.attrs[key] = params_crowns[key]
        
        ########################################### Polygons subgroup
        # Create 'polygons' subgroup
        grp = f.create_group(group_name + '/polygons')

        # Assign polygons as datasets in 'polygons' subgroup
        for idx, poly in crowns['polygons'].items():  
            key = str(idx).zfill(5) # Convert idx to key to string with zero padding
            grp.create_dataset(key, data = poly)

        ########################################### Features subgroup
        # Create 'features' subgroup
        grp = f.create_group(group_name + '/features')

        # Assign all features (i.e. feature sets like 'terrestrial or 'photometric') in crowns['features'] as datasets in 'features' subgroup.
        for key, features in crowns['features'].items():

            # Assert that number of crowns in features are the same as number of crowns in params_crowns
            message = f"`len(crowns['features'][{key}]` is {features.shape[0]} but `params_crowns['number_crowns']` is {params_crowns['number_crowns']})"
            assert features.shape[0] == params_crowns['number_crowns'], message

            # Write features as dataset
            grp.create_dataset(key, data = features)
            
            # Define attributes dict for features dataset
            features_attrs = {}
            features_attrs['names'] = [name for name in features.dtype.names]
            features_attrs['dtypes'] = [str(features.dtype[i]) for i in range(len(features.dtype))]

            # Write feature parameters into main group attributes
            for key_attr, val_attr in features_attrs.items():
                grp_main.attrs['features_' + key + '_' + key_attr] = val_attr

                
    pass

#%%
def load_shapefile(dir_names: List, params_channels: Dict, remove_outliers=True, verbose: bool = True) -> Tuple[Dict, Dict]:

    # Search for paths to wanted files in given directories
    paths = _find_paths_in_dirs(dir_names, params_channels['tnr'])

    # Get affine geo-trafo as rasterio.Affine to convert geo to px coordinates
    affine = transformations.affine_numpy_to_resterio(params_channels['affine'])

    # Open shapefile
    with shapefile.Reader(paths['path_shapes']) as sf:

        # Read dimensions, names, etc.
        n_crowns = len(sf)
        attr_names = [f[0] for f in sf.fields[1:]]

        # Define which attributes will be excluded in final output
        attr_names_exclude = ['Tnr']
        attr_names_include = [name for name in attr_names if name not in attr_names_exclude]
        
        # Init crowns polygons dictionary
        crowns_polys = {}

        # Init crowns records as numpy structed array
        dtype = transformations.geojson_records_fields_to_numpy_dtype(sf.fields, attr_names_exclude)
        
        print(dtype)
        crowns_recs = np.zeros(n_crowns, dtype = dtype)
        

        ##################### Assign polygons and records to crowns
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


    ##################### Remove polygons and corresponding records that lie partially outside image extent or enclose zero area
    shape = params_channels['shape']

    # Init outliers
    outlier_idxs = []
    outlier_keys = []

    # Find outliers
    for idx, (key, poly) in enumerate(crowns_polys.items()):

        # Define bool if poly is completely in image
        is_in_image = (np.min(poly[:, 0]) >= 0) & (np.min(poly[:, 1]) >= 0)  # Lower
        is_in_image = is_in_image & (np.max(poly[:, 0]) < shape[1]) & (np.max(poly[:, 0]) < shape[1])  # Upper (poly is x,y <-> column,row)

        # Define bool if enclosed area greater zero
        area_greater_zero = ( np.max(poly[:, 0]) - np.min(poly[:, 0])) * ( np.max(poly[:, 1]) - np.min(poly[:, 1]))
        area_greater_zero = area_greater_zero > 0

        if (not is_in_image) or (not area_greater_zero):
            outlier_idxs.append(idx)
            outlier_keys.append(key)

    # Remove outliers
    if remove_outliers:
        for key in outlier_keys: 
            crowns_polys.pop(key)
        crowns_recs = np.delete(crowns_recs, outlier_idxs, axis=0)

    # Update the number of crowns
    n_crowns = len(crowns_polys)


    ##################### Prepare return values

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
    params['shape'] = shape
    params['path_shapes'] = paths['path_shapes']
    params['origin'] = 'human'
    params['number_crowns'] = n_crowns

    # Sort parameters according to key
    params = dict([(key, params[key]) for key in sorted(params.keys())])
    
    # Print parameters
    if verbose:
        print('-----------')
        print('Parameters:')
        for k in params: print(f"  {k:<30}: {params[k]}")

    return crowns, params




#%% 
def load_hdf5(
    path: str,
    groups: List = ['channels', 'crowns_human', 'crowns_water'],
    features_only = False,
    ) -> Tuple[Dict, Dict, str]:

    for group in groups:
        assert group in ['channels', 'crowns_human', 'crowns_water'], f"Group `{group}` is not a valid group."
    # Initialize data and parameters dictionary
    data = {}
    params_data = {}
    info = ''

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
                    params_data['channels'] = dict([(key, grp.attrs[key]) for key in grp.attrs.keys()])
                except:
                    info += f"Warning: Group `{group}` not found under path: {path}\n"

            ################################ Crowns
            if bool(re.search('crowns_*', group)):
                
                try:
                    # Get crowns group pointer
                    grp = f.get(group)
                    # Assign `crowns_*` attributes to params as dict
                    params_data[group] = dict([(key, grp.attrs[key]) for key in grp.attrs.keys()])
                    # Initialize crowns sub-dictionary
                    data_crowns = {}
                    
                    ################################ Features sub-group
                    try:
                        # Get features sub-group pointer
                        grp = f.get(group + '/features')
                        # Get features as dict
                        features = dict([(key, grp.get(key)[()]) for key in grp.keys()])
                        # Convert features to numpy arrays with correct dtype
                        for key, val in features.items():
                            features[key] = np.array(val, dtype=val.dtype)
                        # Assign features to crowns sub-dictionary
                        data_crowns['features'] = features
                    except:
                        info += f"Warning: Group `{group + '/features'}` not found under path: {path}\n"

                    ################################ Polygons sub-group
                    # Assign polygons datasets
                    if not features_only:
                        try:
                            # Get polygons sub-group pointer
                            grp = f.get(group + '/polygons')
                            # Get features as dict
                            polygons = dict([(int(key), grp.get(key)[()]) for key in grp.keys()])
                            # Assign polygons to crowns sub-dictionary
                            data_crowns['polygons'] = polygons
                        except:
                           exceptions += f"Warning: Group `{group + '/polygons'}` not found under path: {path}\n"
                    
                    # Assign crowns sub-dictionary to data
                    data[group] = data_crowns
                    
                    # Add info to params
                    params_data['io.load_hdf5()_info'] = info

                except:
                    info += f"Warning: Group `{group}` not found under path: {path}\n"
                
    return data, params_data


#%%
def allhdf5s_crowns_features_to_dataframe(
    dir_hdf5s: str,
    crowns_type = 'crowns_human',
    ) -> pd.DataFrame:
    
    assert crowns_type in ['crowns_human', 'crowns_water'], f"`{crowns_type}` is not a valid crowns_type."
    
    # Get paths to all available hdf5 files
    paths = [os.path.join(dir_hdf5s, name) for name in os.listdir(dir_hdf5s) if os.path.splitext(name)[-1] == '.hdf5']

    params_features = {}
    features_terr = {}
    features_photo = {}

    for path in paths:

        # Load crowns_type group features only in each hdf5
        data, params_data = load_hdf5(path, groups = [crowns_type], features_only=True)
        
        # Try to get tnr and crowns parameters
        try:
            tnr = int(params_data[crowns_type]['tnr'])
            params_features[tnr] = params_data[crowns_type]
        except:
            pass

        # Try to add photometric & terrestrial features to respective dict
        try:
            features_photo[tnr] = pd.DataFrame( data[crowns_type]['features']['photometric'] ).assign(tnr = tnr)
        except:
            pass
        try:
            features_terr[tnr] = pd.DataFrame( data[crowns_type]['features']['terrestrial'] ).assign(tnr = tnr)
        except:
            pass
    
    # Collect info for which tnr there are no combined photometric and terrestrial features
    info = []
    for key in features_terr:
        if key not in features_photo.keys():
            info.append( f"tnr{key}: Only terr. features." )
    for key in features_photo:
        if key not in features_terr.keys():
            info.append( f"tnr{key}: Only photo. features." )
    
    params_features['io.allhdf5s_crowns_features_to_dataframe()_info'] = info

    # Combine terrestrial features
    features_terr_df = pd.concat([f for f in features_terr.values()], axis = 0)
    features_terr_df.reset_index(inplace=True, drop=True)

    # Combine photometric features
    features_photo_df = pd.concat([f for f in features_photo.values()], axis = 0)
    features_photo_df.reset_index(inplace=True, drop=True)

    # Now merge terrestrial & photometric features on [tnr, id]
    features = pd.merge(
        features_terr_df,
        features_photo_df,
        on = ['tnr','id'],
        how='inner',
    )


    return features, params_features