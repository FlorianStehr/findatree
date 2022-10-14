from typing import List, Tuple, Dict
import numpy as np
import warnings
import importlib

import skimage.filters
import skimage.morphology
import skimage.util

import findatree.transformations as transformations

importlib.reload(transformations)


def _mask_global_otsu(
    channels: Dict,
    channel_name: str = 'avg',
    thresh_height_lower: float = 5,
    thresh_height_upper: float = 40,
    ) -> np.ndarray:
    """Calculate shadow mask using global otsu thresholding on specified channel.

    Parameters
    ----------
    channels : dict
        Channels dictionary see `findatree.geo_to_image.channels_load()`.
    channel_name: str
        Name of channelthat is used for thresholding.
    thresh_height_lower:
        Pixel values with chm below this value are not taken into account during otsu thresholding.
    thresh_height_upper:
        Pixel values with chm above this value are not taken into account during otsu thresholding.

    Returns
    -------
    np.ndarray
        Shadow mask of type numpy.uint8 with foreground -> 1 (e.g. no shadow) and background -> 0 (e.g. shadow).
    """
   
    # Extend channels when mean over all channels is not present
    if channel_name not in channels.keys():
        transformations.channels_extend(
            channels,
            extend_by=[channel_name],
            )
      
    # Mean image and values (e.g. flattened image)  
    img = channels[channel_name].copy()
    
    # Prepare mask for values ctaken into account during otsu thresholding
    mask_otsu = np.isfinite(img) & (channels['chm'] > thresh_height_lower) & (channels['chm'] < thresh_height_upper)
    
    # Calculate Otsu global otsu threshold
    thresh = skimage.filters.threshold_otsu(img[mask_otsu])
    
    # Set mask, i.e. background -> 0, foreground -> 1
    mask = img.copy()
    mask[mask < thresh] = 0
    mask[mask >= thresh] = 1
    mask = mask.astype(np.bool)
    
    return mask
    


def _mask_local_otsu(
    channels: Dict,
    channel_name: str = 'avg',
    width: int = 101,
    thresh_height_lower: float = 5,
    thresh_height_upper: float = 40,
    ) -> np.ndarray:
    """Calculate shadow mask using local otsu thresholding on specified channel.

    Parameters
    ----------
    channels : dict
        Channels dictionary see `findatree.geo_to_image.channels_load()`.
    channel_name: str
        Name of channelthat is used for thresholding.
    width: int
        Width of square footprint (e.g. sliding window) in px used for local otsu thresholding.
    thresh_height_lower:
        Pixel values with chm below this value are not taken into account during otsu thresholding.
    thresh_height_upper:
        Pixel values with chm above this value are not taken into account during otsu thresholding.
        
    Returns
    -------
    np.ndarray
        Shadow mask of type numpy.uint8 with foreground -> 1 (e.g. no shadow) and background -> 0 (e.g. shadow).
    """
   
    # Extend channels when mean over all channels is not present
    if channel_name not in channels.keys():
        transformations.channels_extend(
            channels,
            extend_by=[channel_name],
            )
      
    # Mean image and values (e.g. flattened image)  
    img = channels[channel_name].copy()
    vals = img.flatten()
    
    # Get minimum and maximum values
    perc = 1
    img_min = np.nanpercentile(vals, perc)
    img_max = np.nanpercentile(vals, 100 - perc)
    
    # Rescale image to 8bit imagae
    img_u8 = skimage.exposure.rescale_intensity(
            img,
            in_range=(img_min, img_max),
            out_range=(0,255),
        ).astype(np.uint8)
    
    # Prepare mask for values ctaken into account during otsu thresholding
    mask_otsu = np.isfinite(img) & (channels['chm'] > thresh_height_lower) & (channels['chm'] < thresh_height_upper)
    
    # Compute local otsu threshold 
    thresh_local = skimage.filters.rank.otsu(
        img_u8,
        skimage.morphology.square(width),
        mask = mask_otsu, 
    )
    
    # Set mask, i.e. background -> 0, foreground -> 1
    mask = img_u8.copy()
    mask[mask < thresh_local] = 0
    mask[mask >= thresh_local] = 1
    mask = mask * mask_otsu.astype(np.uint8)
    mask = mask.astype(np.bool)
    
    return mask