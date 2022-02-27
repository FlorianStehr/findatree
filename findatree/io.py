from typing import List
from typing import Tuple

import numpy as np

import rasterio
from rasterio.warp import reproject, Resampling

import cv2 as cv

import importlib

from findatree import segmentation as segment

importlib.reload(segment)

#%%
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
def reproject_all_intersect(paths: List[str], res: float) -> Tuple:
    '''
    Reproject all rasters in raster-files (.tif) given by ``paths`` to the intersection of all rasters and a defined resolution ``res`` using rasterio package.
    
    Parameters:
    ----------
    paths: List[str]
        Full path names to raster-files
    res: float
        Resolution per px in final rasters.

    Returns:
    -------
    dest_bands_list: List[np.ndarray]
        List of reprojected rasters in intersection and with defined resolution ``res`` given as np.ndarray of dtype=np.float64.
    dest_mask: np.ndarray
        Intersection mask of all repojected rasters as np.ndarray. Valid values -> 1, Non-valid values -> np.nan.
    dest_A: rasterio.Affine
        Affine geo-transform (rasterio) of all reprojected rasters
    inter_bound: dict
        Bounds in geo-coordinates of intersection box.
    
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
        int(np.floor((inter_bound['top'] - inter_bound['bottom']) / res)),
        int(np.floor((inter_bound['right'] - inter_bound['left']) / res)),
    )
    
    # Define common (i.e. destination) Affine based on intersection area and resolution
    dest_A = rasterio.Affine(
        res, 0, inter_bound['left'],
        0, -res, inter_bound['top'])

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
    
    return dest_bands_list, dest_mask, dest_A, inter_bound

#%%

def _close_nan_holes(img: np.ndarray, max_pxs: int = 10**2) -> np.ndarray:
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

    # Get indices of connectedComponents (ccs)
    ccs_idx = segment._connectedComponents_idx(mask)

    # Remove all ccs with len(cc) > max_pxs
    ccs_idx = [cc_idx for cc_idx in ccs_idx if len(cc_idx[0]) <= max_pxs]

    # Define kernel for dilation of ccs
    kernel = np.ones((3,3), dtype=np.uint8)

    for i, cc_idx in enumerate(ccs_idx):
        # Bounding box for cc
        box_lims = [
            (np.min(cc_idx[0]) - 1, np.max(cc_idx[0]) + 2),
            (np.min(cc_idx[1]) - 1, np.max(cc_idx[1]) + 2),
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
        img_closed[cc_idx] = np.nanmedian(img_closed[cc_ring_idx])

        # Assign val of 2 to ring values in mask
        mask[cc_ring_idx] = 2

    return img_closed, mask