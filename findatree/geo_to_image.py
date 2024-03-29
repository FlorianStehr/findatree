from typing import Dict, List, Tuple
import numpy as np
import importlib
import rasterio
from rasterio.warp import reproject, Resampling
import cv2 as cv
import skimage.measure as measure

# Import findatree modules
from findatree import io as io
from findatree import transformations as transformations

importlib.reload(transformations)
importlib.reload(io)


#%%
def print_raster_info(paths: List[str]) -> None:
    """Quickly print some information about .tif geo-raster-file defined by paths.

    Parameters
    ----------
    paths : List[str]
        List of absolute paths to raster files.
    """
    for i, path in enumerate(paths):
        with rasterio.open(path) as ds:
            print()
            print(f"({i})")
            print(f"Name: {ds.name}")
            print(f"Width[px]: {ds.width}")
            print(f"Height[px]: {ds.height}")

            print(f"No. of rasters: {ds.count}")
            for i, dtype in zip(ds.indexes, ds.dtypes):
                print(f"  Index: {i}, dtype: {dtype}")
            
            print(f"Nodata values: {ds.nodatavals}")
            
            print()
            print(f"Coordinate reference system CRS: {ds.crs}")
            print(f"Geo bounds: {ds.bounds}")
            print(f"Affine geo-transfrom: {[val for val in ds.transform[:-3]]}")

    
#%%
def _reproject_to_primary(paths_dict: Dict, px_width: float) -> Tuple[Dict, Dict]:
    """Reproject all rasters in dsm, dtm and ortho raster-files given by `paths` to the intersection of all rasters with same resolution given by `px_width` using rasterio package.
    
    We refer to these reprojected raster files as primary channels.

    Parameters:
    ----------
    paths_dict: Dict
        Dictionary of full path names to raster-files.Must contain keys `path_dsm`, `path_dtm`, `path_ortho`, see findatree.io._find_paths_in_dirs()
    px_width: float
        Isotropic width per pixel in [m] of all reprojected rasters/primary channels.

    Returns:
    -------
    Tuple[Dict, Dict]
        cs_prim: dict[np.ndarray,...]
            Dictionary of reprojected rasters, i.e. primary channels, in intersection and with defined resolution given by `px_width` given as np.ndarray of dtype=np.float64.
            Keys are: ['dsm', 'dtm', 'blue', 'green', 'red', 're', 'nir'].
        params: dict
        * crs [str]: Coordinate system of all reprojected rasters.
        * affine [np.ndarray]: Affine geo. transform of all reprojected rasters (see rasterio) as np.ndarray
        * px_width_reproject[float]: Isotropic width per pixel in [m] of all reprojected rasters.
        * shape [Tuple]: Shape  in [px] of images of all reprojected rasters.
        * bound_left [float]: Left boundary of intersection of all reprojected rasters (maximum).
        * bound_bottom [float]: Bottom boundary of intersection of all reprojected rasters (maximum).
        * bound_right [float]: Right boundary of intersection of all reprojected rasters (minimum).
        * bound_top [float]: Top boundary of intersection of all reprojected rasters (minimum).

    Notes:
    -----
    * Besides bounding box also the masks of all rasters are extracted and the interesction mask (``dest_mask``) is computed.
    * All values outside of intersection mask are assigned with numpy.nan values.
    * All fully saturated values for specific dtype of original rasters are assigned with numpy.nan values.

    """
    # Assert that all the path keys are in paths_dict
    path_keys = ['path_dsm', 'path_dtm', 'path_ortho']
    for key in path_keys:
        assert key in paths_dict, f"Key `{key}` missing in `paths_dict`"

    # Convert paths_dict to list of paths in correct order
    paths = [paths_dict[key] for key in path_keys]

    # Load bounds and CRSs of all raster-files
    bounds = []
    crss = []
    for i, path in enumerate(paths):
        with rasterio.open(path) as ds:
            bounds.append(ds.bounds)
            crss.append(ds.crs)
    
    # Define intersection area of all rasters
    inter_bound = {
        'left': 0,
        'bottom': 0,
        'right': 0,
        'top': 0,
    }

    inter_bound['left'] = np.max([b.left for b in bounds])
    inter_bound['bottom'] = np.max([b.bottom for b in bounds])
    inter_bound['right'] = np.min([b.right for b in bounds])
    inter_bound['top'] = np.min([b.top for b in bounds])
    
    # Define common (i.e. destination) shape in px of all rasters after reprojection based on intersection area and resolution
    dest_shape = (
        int(np.floor((inter_bound['top'] - inter_bound['bottom']) / px_width)),
        int(np.floor((inter_bound['right'] - inter_bound['left']) / px_width)),
    )
    
    # Define common (i.e. destination) Affine based on intersection area and resolution
    dest_A = rasterio.Affine(
        px_width, 0, inter_bound['left'],
        0, -px_width, inter_bound['top'])

    # Define common (i.e. destination) coordinatereference system 
    dest_crs = crss[0]

    # Reproject rasters and masks
    dest_bands_list = []
    dest_mask = np.ones((dest_shape[0], dest_shape[1])) # This will be the intersection of all masks

    for path in paths:

        with rasterio.open(path) as ds:  # Open raster-file
            source_crs = ds.crs
            source_A = ds.transform

            dest_bands = np.zeros((dest_shape[0], dest_shape[1], ds.count))
            
            for i_raster, index in enumerate(ds.indexes):  # Cycle through bands
                ##################### Band
                # Reproject band
                source_band = ds.read(index)
                source_dtype = ds.dtypes[i_raster]

                dest_band = np.zeros((dest_shape[0], dest_shape[1]))

                reproject(
                    source_band,
                    dest_band,
                    src_transform=source_A,
                    src_crs=source_crs,
                    dst_transform=dest_A,
                    dst_crs=dest_crs,
                    dst_nodata=0,
                    resampling=Resampling.nearest
                )
                
                # Handle minimal/maximal values of each dtype
                try:
                    dest_band[dest_band.astype(source_dtype) == np.iinfo(source_dtype).min] = 1
                    dest_band[dest_band.astype(source_dtype) == np.iinfo(source_dtype).max] = 1
                except:
                    dest_band[dest_band.astype(source_dtype) == np.finfo(source_dtype).min] = 0

                # Assign band to bands
                dest_bands[:,:,i_raster] = dest_band

                ##################### Mask
                # Reproject mask
                source_band_mask = ds.read_masks(index)
                dest_band_mask = np.zeros((dest_shape[0], dest_shape[1]))
            
                reproject(
                    source_band_mask,
                    dest_band_mask,
                    src_transform=source_A,
                    src_crs=source_crs,
                    dst_transform=dest_A,
                    dst_crs=dest_crs,
                    dst_nodata=0,
                    resampling=Resampling.nearest
                )
                
                # Assign NaNs to nodata values of mask
                dest_band_mask[dest_band_mask == 0] = np.nan

                # Check if whole raster is empty
                isempty = np.sum(np.isnan(dest_band).flatten()) == (dest_shape[0]*dest_shape[1])

                # If raster is not completely empty combine raster with mask
                if not isempty:
                    dest_band_mask = dest_band_mask * dest_band

                # Set data value to 1
                dest_band_mask[np.isfinite(dest_band_mask)] = 1

                # Combine to intersection mask, nodata -> NaN, data -> 1
                dest_mask = dest_mask * dest_band_mask

        # Add bands to list of bands
        dest_bands_list.append(dest_bands)
    
    # Now go through all reprojected rasters and apply dest_mask
    dest_bands_list = [dest_bands * dest_mask.reshape((dest_shape[0], dest_shape[1], 1)) for dest_bands in dest_bands_list]

    # Return primary channels as dictionary
    cs_prim = {}
    cs_prim['mask'] = dest_mask
    for i, band in enumerate(dest_bands_list):
        if i==0:
            cs_prim['dsm'] = band[:,:,0]
        elif i==1:
            cs_prim['dtm'] = band[:,:,0]
        elif i==2:
            cs_prim['blue'] = band[:,:,0]
            cs_prim['green'] = band[:,:,1]
            cs_prim['red'] = band[:,:,2]
            cs_prim['re'] = band[:,:,3]
            cs_prim['nir'] = band[:,:,4]
    
    # Return reprojection parameters as dictionary
    params = {}
    params['crs'] = str(dest_crs)
    params['affine'] = np.array(dest_A).reshape((3,3))
    params['px_width_reproject'] = px_width
    params['shape'] = dest_shape
    params['bound_left'] = float(inter_bound['left'])
    params['bound_bottom'] = float(inter_bound['bottom'])
    params['bound_right'] = float(inter_bound['right'])
    params['bound_top'] = float(inter_bound['top'])


    return cs_prim, params


#%%
def _channels_normalize(
    channels_reproject: Dict,
    ) -> Tuple:
    """Normalize/convert reprojected rasters (primary channels).

    Parameters
    ----------
    channels_reproject : Dict
        Dictionary of of reprojected primary channels as np.ndarray of dtype=np.float64, as returned by geo_to_image._reproject_to_primary().
        Must contain following channels/keys `'blue', 'green', 'red', 're', 'nir', 'dsm', 'dtm'`.
        
    Returns
    -------
    Dict
        Dict. of normalized reprojected channels as np.ndarray of type np.float32.
        Keys are: ['blue', 'green', 'red', 're', 'nir', 'chm'].
            
    Notes:
    ------
    Color channels `['blue', 'green', 'red', 're', 'nir']` are normalized to interval [0, 2].
    Normalization constant  for color channels is `2^15 - 1` according to Agisoft standards, 
    where maximum reflection of 1 occurs at this value.
    I.e. also values > 1 can occur after normalization if calibration was not sucsessful.
    The ratio of maxed out channels to the total number of finite pixels (key: `'max_ratio_*'`) is saved in the output parameters dictionary for each color channel.
    
    Canopy height model `chm` is normalized to interval [0, inf[.
    

    """
    channels = {}

    ############################## Primary channel normalization
    # Normalize and convert to float32 dtype
    norm_const = 2**15 - 1
    channels['blue'] = (channels_reproject['blue'] / norm_const).astype(np.float32)
    channels['green'] = (channels_reproject['green'] / norm_const).astype(np.float32)
    channels['red'] = (channels_reproject['red'] / norm_const).astype(np.float32)
    channels['re'] = (channels_reproject['re'] / norm_const).astype(np.float32)
    channels['nir'] = (channels_reproject['nir'] / norm_const).astype(np.float32)


    ############################## CHM
    channels['chm'] = (channels_reproject['dsm'] - channels_reproject['dtm']).astype(np.float32)

    # Set all below zero values to zero
    for key in channels:
        img = channels[key]
        img[img <= 0] = 0
        
       
    ############################## Get ratio of finite pixels to maxed out pixels (>1)
    # Total number of finite pixels
    n_px_finite = np.sum(np.isfinite(channels['blue']).flatten())
    
    # Define color channels
    color_names = ['blue', 'green', 'red', 're', 'nir']
    
    # Loop through color channels and compute ratio of maxed out pixels
    params = {}
    for name in color_names:
        n_px_max = np.sum((channels[name] >= 1).flatten())
        params['max_ratio_' + name] = n_px_max / n_px_finite

    return channels, params


#%%
def _channels_downscale(
    channels_in: Dict,
    params_in: Dict,
    downscale: int = 0,
    ) -> Tuple[Dict, Dict]:
    """Downscale secondary channels by using gaussian image pyramids

    Parameters
    ----------
    channels_in: Dict
        Dictionary of channels as np.ndarray of dtype=np.float32.
    params_in: Dict
        Dictionary of parameters of channels, must contain key `'px_width_reproject'`.
    downscale : int, optional
        Pixel downscale factor by using gaussian image pyramids, by default 0.

    Returns
    -------
    Tuple[Dict, Dict]
        channels : Dict
            Dictionary of downscaled channels as np.ndarray of dtype=np.float32.
        params : Dict
        * px_width [float]: Final ajusted pixel width after downscaling in meters.
        * affine [np.ndarray]: Affine geo. transform after downscaling.
        * shape [Tuple]: Shape of image in pixels after downscaling.
    """
    channels = channels_in.copy()
    params = params_in.copy()

    if downscale > 0:
        # Loop through every channel
        for key in channels:
            img = channels[key].copy()
            # Gaussian image pyramid
            for i in range(downscale):
                img = cv.pyrDown(img)
            channels[key] = img
        # Get shape
        shape = img.shape
    else:
        # Get (unchanged) shape
        shape = params['shape']

    params['px_width'] = params['px_width_reproject'] * 2**downscale
    params['shape'] = shape
    params['affine'][0,0] = params['px_width']
    params['affine'][1,1] = - params['px_width']

    return channels, params


#%%
def channels_load(
    dir_names: List[str],
    params: Dict,
    verbose: bool = True,
    ) -> Tuple[Dict, Dict]:
    """Reproject dsm, dtm and ortho rasters to of same area code to same intersection and resolution and convert/normalize to secondary channels.

    This function combines:
    * findatree.io._find_paths_in_dirs()
    * findatree.geo_to_image._reproject_to_channels()
    * findatree.geo_to_image._channels_primary_to_secondary() 
    * findatree.geo_to_image._channels_downscale()

    Parameters
    ----------
    dir_names : List[str]
        List of absolute paths to all folders containing the necessary dsm, dtm and ortho rasters.
    params : Dict
        * tnr [int]: Area identifier.
        * px_width_reproject [float]: Reprojection pixel width in meters, by default 0.2.
        * downscale [int]: Additionaly downscale px_width after reprojection by factor `2**(downscale)` using gaussian image pyramids.
    verbose: bool
        Print parameters at end of execution, by default True.
        
    Returns
    -------
    Tuple[Dict, Dict]
        channels: Dict[np.ndarray, ...]
            Normalized and downscaled channels as np.ndarray of type np.float32.
            Keys are: ['blue', 'green', 'red', 're', 'nir', 'chm'].
        params: Dict
        * date_time [str]: Processing date and time
        * tnr [int]: Area identifier
        * path_dsm [str]: Absolute path to dsm raster
        * path_dtm [str]: Absolute path to dtm raster
        * path_ortho [str]: Absolute path to ortho raster
        * crs [str]: Coordinate system of all reprojected rasters. 
        * px_width_reproject[float]: Isotropic width per pixel in [m] of all reprojected rasters, i.e. primary channels.
        * downscale [int]: Downscale factor of pixel size by uisng gaussian image pyramids, by default 0.
        * px_width[float]: Isotropic width per pixel in [m] of all secondary (downscaled) channels.
        * shape [Tuple]: Final shape  in [px] of all secondary channels.
        * affine [np.ndarray]: Affine geo. transform of all of secondary channels (see rasterio) as np.ndarray.

    """
    ######################################### (0) Set standard settings if not set
    params_standard = {
        'tnr': None,
        'px_width_reproject': 0.2,
        'downscale': 0,
    }
    for k in params_standard:
        if k in params:
            params[k] = params[k]
        else:
            params[k] = params_standard[k]


    ######################################### (1) Function calls

    # Find paths to dsm, dtm, ortho rasters of same area code
    paths_dict = io._find_paths_in_dirs(dir_names, tnr_number=params['tnr'])   
    
    # Reproject dsm, dtm, ortho rasters to same resolution and intersection -> primary channels
    channels_reprojected, params_reprojected, = _reproject_to_primary(paths_dict, px_width=params['px_width_reproject'])
    
    # Normalize reprojected channels
    channels_normalized, params_normalized = _channels_normalize(channels_reprojected)

    # Downscale normalized channels
    channels_downscaled, params_downscaled = _channels_downscale(channels_normalized, params_reprojected, downscale=params['downscale'])
    
    # Final output
    channels = channels_downscaled

    ######################################### (2) Prepare parameters

    # Add all path params to final params
    for key in paths_dict:
        params[key] = paths_dict[key]
    
    # Add date of processing to params
    params['date_time'] = transformations.current_datetime()

    # Add missing geo-referencing params
    params['crs'] = params_reprojected['crs']
    params['affine'] = params_downscaled['affine']
    params['px_width'] = params_downscaled['px_width']
    params['shape'] = params_downscaled['shape']
    
    # Add maxed out pixel ratios from normalization to params
    for key, val in params_normalized.items():
        params[key] = val
    
    # Sort parameters according to key
    params = dict([(key, params[key]) for key in sorted(params.keys())])

    # Print parameters
    if verbose:
        print('-----------')
        print('Parameters:')
        for k in params: print(f"  {k:<30}: {params[k]}")
    
    return channels, params