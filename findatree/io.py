from typing import Dict, List, Tuple
import glob
import re
import os
import numpy as np
import importlib
import rasterio
from rasterio.warp import reproject, Resampling
import cv2 as cv
import skimage.measure as measure
import skimage.exposure as exposure

from findatree import object_properties as objprops

importlib.reload(objprops)

   
#%%
def check_path(paths, pattern):
    if len(paths) == 0:
        path = None
        print('0 '+ pattern + 's found in given directories!!!')
        print('  -> Returning ' + pattern + ': ' + 'None')
    elif len(paths) == 1:
        path = paths[0]
        print('1 '+ pattern + 's found in given directories, OK.')
        print('  -> Returning ' + pattern + ': ' + os.path.split(path)[-1])

    else:
        path = paths[0]
        print(str(len(paths))+ ' ' + pattern + 's found in given directories!!!')
        print('  -> Returning ' + pattern + ': ' + os.path.split(path)[-1])
    
    return path

#%%
def find_paths_in_dirs(
    dir_names,
    tnr_number=None,
    verbose=True,
    filetype_pattern = '*.tif',
    dsm_pattern = 'dsm',
    dtm_pattern = 'dtm',
    ortho_pattern = 'ortho',
    ):

    # Get full paths to all files that match ``filetype_pattern`` in all directories
    paths = []
    for dir_name in dir_names:
            path = sorted(glob.glob( os.path.join( dir_name,filetype_pattern) ) )
            paths.extend(path)

    # ``tnr_number`` to pattern 
    if tnr_number is None:
        print('No tnr found, displaying all available ' + filetype_pattern + 's:')
        for p in paths: print('  ' + p)
        print()

    tnr_number = str(tnr_number)
    tnr_pattern = 'tnr_' + tnr_number

    # Get paths that contain ``tnr_number``
    paths = [p for p in paths if bool(re.search(tnr_pattern, os.path.split(p)[-1], re.IGNORECASE))]

    # Get path that contains non-case sensitive ``dsm_pattern``
    paths_dsm = [p for p in paths if bool(re.search(dsm_pattern, os.path.split(p)[-1], re.IGNORECASE))]
    path_dsm = check_path(paths_dsm, dsm_pattern)

    # Get path that contains non-case sensitive ``dtm_pattern``
    paths_dtm = [p for p in paths if bool(re.search(dtm_pattern, os.path.split(p)[-1], re.IGNORECASE))]
    path_dtm = check_path(paths_dtm, dtm_pattern)

    # Get path that contains non-case sensitive ``ortho_pattern``
    paths_ortho = [p for p in paths if bool(re.search(ortho_pattern, os.path.split(p)[-1], re.IGNORECASE))]
    path_ortho = check_path(paths_ortho, ortho_pattern)

    # Join dsm/dtm/ortho paths
    paths = [
        path_dsm,
        path_dtm,
        path_ortho,
    ]

    params = {
        'tnr': tnr_number,
        'path_dsm': paths[0],
        'path_dtm': paths[1],
        'path_ortho': paths[2],
    }

    # Print parameters
    if verbose:
        print('-----------')
        print('Parameters:')
        for k in params: print(f"  {k:<30}: {params[k]}")


    return paths, params

##%%
def print_raster_info(paths: List[str]) -> None:
    '''
    Quickly print some information about .tif geo-raster-file defined by paths.
    '''
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
def reproject_all_intersect(paths: List[str], px_width: float, verbose: bool=True) -> Tuple[Dict, Dict]:
    '''
    Reproject all rasters in raster-files (.tif) given by ``paths`` to the intersection of all rasters and a defined resolution ``res`` using rasterio package.
    
    Parameters:
    ----------
    paths: List[str]
        Full path names to raster-files in order [``path_dsm``, ``path_dtm``, ``path_ortho``]
    resolution: float
        Resolution per px in final rasters.
    verbose: bool, optional
        Print function parameters, by default True

    Returns:
    -------
    Tuple[Dict, Dict]
        cs_prim: dict[np.ndarray,...]
            Dictionary of reprojected rasters in intersection and with defined resolution ``resolution`` given as np.ndarray of dtype=np.float64.
            Keys are: ['dsm', 'dtm', 'blue', 'green', 'red', 're', 'nir'].
        params: dict
        * crs [str]: Coordinate system of all reprojected rasters.
        * affine [np.ndarray]: Affine geo. transform of all reprojected rasters (see rasterio) as np.ndarray
        * px_width[float]: Isotropic width per pixel in [m] of all reprojected rasters.
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

    '''
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
                # Assing NaNs to saturated values within source dtype
                try:
                    dest_band[dest_band.astype(source_dtype) == np.iinfo(source_dtype).max] = np.nan
                    dest_band[dest_band.astype(source_dtype) == np.iinfo(source_dtype).min] = np.nan
                except:
                    dest_band[dest_band.astype(source_dtype) == np.finfo(source_dtype).max] = np.nan
                    dest_band[dest_band.astype(source_dtype) == np.finfo(source_dtype).min] = np.nan
                
                # Close small NaN holes in band
                dest_band = _close_nan_holes(dest_band)[0]

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
    params['px_width'] = px_width
    params['shape'] = dest_shape
    params['bound_left'] = float(inter_bound['left'])
    params['bound_bottom'] = float(inter_bound['bottom'])
    params['bound_right'] = float(inter_bound['right'])
    params['bound_top'] = float(inter_bound['top'])

    # Print parameters
    if verbose:
        print('-----------')
        print('Parameters:')
        for k in params: print(f"  {k:<30}: {params[k]}")


    return cs_prim, params

#%%

def _close_nan_holes(img: np.ndarray, max_pxs: int = 200) -> np.ndarray:
    '''
    Close small NaN holes in image with medium value of surroundings.

    Paramaters:
    -----------
    image: np.ndarray
        Image of size (m,n)
    max_pxs: int=10**2
        Only holes of ```len(hole.flatten()) <= max_pxs`` are closed.

    Returns:
    -------
    img_closed:
        Copy of original image with closed NaN holes.
    mask:
        Mask with: 
            * NaN-holes -> 1 
            * Surrounding -> 2
            * All other -> 0
    '''

    # Init return image
    img_closed = img.copy()

    # Create binary mask where pixels of value: NaN -> 1, other -> 0
    mask = np.zeros(img.shape, dtype=np.uint8)
    mask[np.isnan(img)] = 1

    # Label connected components (ccs)
    labels = measure.label(mask, background=0, return_num=False, connectivity=1)

    # Get indices of ccs 
    ccs_idx = objprops.labels_idx(labels)

    # Remove all ccs with len(cc) > max_pxs
    ccs_idx = [cc_idx for cc_idx in ccs_idx if len(cc_idx[0]) <= max_pxs]

    # Define kernel for dilation of ccs
    kernel = np.ones((3,3), dtype=np.uint8)

    for i, cc_idx in enumerate(ccs_idx):
        # Bounding box for cc
        # Also check if bounding box limits are not outside of image shape
        box_lims = [
            (
                max(np.min(cc_idx[0]) - 1, 0),
                min(np.max(cc_idx[0]) + 2, img.shape[0]),
            ),
            (
                min(np.min(cc_idx[1]) - 1, 0),
                max(np.max(cc_idx[1]) + 2, img.shape[1]),
            ),
        ]
        
        # Create small image of cc in box
        cc_box = mask[box_lims[0][0]:box_lims[0][1], box_lims[1][0]:box_lims[1][1]]

        # Dilate cc in box
        cc_box_dilate = cv.dilate(cc_box, kernel)
        
        # Difference between dilated and original cc, i.e. a one pixel ring around the cc
        cc_box_ring = cc_box_dilate - cc_box
        
        # Get box-indices of difference
        cc_box_ring_idx = np.where(cc_box_ring == 1)

        # Convert box_indices back to original mask coordinates
        cc_ring_idx = (cc_box_ring_idx[0] + box_lims[0][0], cc_box_ring_idx[1] + box_lims[1][0])

        # Fill hole in image with median of ring around hole
        try:
            img_closed[cc_idx] = np.nanmedian(img_closed[cc_ring_idx])
        except:
            img_closed[cc_idx] = np.nan

        # Assign val of 2 to ring values in mask
        mask[cc_ring_idx] = 2

    return img_closed, mask


#%%

def channels_primary_to_secondary(cs_prim, params_cs_prim, downscale=0, verbose=True) -> Tuple[Dict, Dict]:
    """channels_primary_to_secondary _summary_

    Parameters
    ----------
    cs_prim : _type_
        Dictionary of of primary channels as np.ndarray of dtype=np.float64, see io.reproject_all_intersect().
    params_cs_prim : _type_
        Dictionary of parameters of primary channels, see io.reproject_all_intersect().
    downscale : int, optional
        Pixel downscale factor, by default 0
    verbose : bool, optional
        Print parameters during call, by default True

    Returns
    -------
    Tuple[Dict, Dict]
        channels: Dict[np.ndarray, ...]
            Dict. of all normalized primary/sesondary channels as np.ndarray of type np.float32.
            Keys are: ['blue', 'green', 'red', 're', 'nir', 'chm', 'ndvi', 'ndvire', 'ndre', 'RGB', 'rgb', 'h', 'l', 's'].
            
        params: Dict
        * blue_wavelength [int]: Wavelength of ... channel
        * green_wavelength [int]: Wavelength of ... channel
        * red_wavelength [int]: Wavelength of ... channel
        * re_wavelength [int]: Wavelength of ... channel
        * nir_wavelength [int]: Wavelength of ... channel
        * downscale [int]: Downscale factor of pixel size, by default 0.
        * px_width [float]: Final pixel width in meters.
        * shape [Tuple]: Final shape of image in pixels.
    """
    cs_prim_names = list(cs_prim.keys())
    shape_prim = params_cs_prim['shape']
    
    sec_in_prim = 'ndvi' in cs_prim_names
    if sec_in_prim: print('    ... [io.channels_primary_to_secondary()] already secondary channels!')

    if not sec_in_prim: # Input is indeed a primary channel dictionary -> Normalize and compute secondary channels!
        channels = {}
        ############################## Primary channel normalization
        # Normalize and convert to float32 dtype
        channels['blue'] = (cs_prim['blue'] / (2**16 - 1)).astype(np.float32)
        channels['green'] = (cs_prim['green'] / (2**16 - 1)).astype(np.float32)
        channels['red'] = (cs_prim['red'] / (2**16 - 1)).astype(np.float32)
        channels['re'] = (cs_prim['re'] / (2**16 - 1)).astype(np.float32)
        channels['nir'] = (cs_prim['nir'] / (2**16 - 1)).astype(np.float32)


        ############################## Secondary channels
        # Canopy height model (CHM)
        channels['chm'] = (cs_prim['dsm'] - cs_prim['dtm']).astype(np.float32)

        # Vegetation indices
        channels['ndvi'] = (channels['nir'] - channels['red']) / (channels['nir'] + channels['red'])
        channels['ndvire'] = (channels['re'] - channels['red']) / (channels['re'] + channels['red'])
        channels['ndre'] = (channels['nir'] - channels['re']) / (channels['nir'] + channels['re'])

        # RGB
        rgb = np.zeros((shape_prim[0], shape_prim[1], 3), dtype=np.float32)
        rgb[:,:,0] = channels['red']
        rgb[:,:,1] = channels['green']
        rgb[:,:,2] = channels['blue']
        channels['RGB'] = rgb  # Three channel RGB image
        channels['rgb'] = np.mean(rgb, axis=2)  # Arithmetic mean RGB image

        # HLS
        hls = cv.cvtColor(rgb, cv.COLOR_RGB2HLS)
        channels['h'] = hls[:,:,0]
        channels['l'] = hls[:,:,1]
        channels['s'] = hls[:,:,2]

    else: # Input is already normalized -> Just copy input to channels!
        channels = cs_prim.copy()

    # Return parameters as dictionary
    params = {}
    params['blue_wavelength'] = 450
    params['green_wavelength'] = 560
    params['red_wavelength'] = 650
    params['re_wavelength'] = 730
    params['nir_wavelength'] = 840

    params['downscale'] = downscale
    params['px_width'] = params_cs_prim['px_width'] * 2**downscale

    # Optional resolution reduction by gaussian image pyramid
    if downscale == 0:
        params['shape'] = params_cs_prim['shape']
    
    else:
        for key in channels:
            img = channels[key].copy()
            for i in range(downscale):
                img = cv.pyrDown(img)
            channels[key] = img
        shape_sec = channels[list(channels.keys())[0]].shape[:2]
        params['shape'] = shape_sec
    
    # Print parameters
    if verbose:
        print('-----------')
        print('Parameters:')
        for k in params: print(f"  {k:<30}: {params[k]}")

    return channels, params