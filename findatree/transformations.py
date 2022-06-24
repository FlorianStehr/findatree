from typing import Dict, List, Tuple
import numpy as np
import cv2
import skimage.exposure

def rgb_to_RGBA(red:np.ndarray, green: np.ndarray, blue: np.ndarray, perc: float = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Convert red, green and blue float32 images of shape (M,N) to RGB uint8 image of shape (M,N,3) and RGBA uint32 image of shape (M,N).

    Parameters
    ----------
    red: np.ndarray
        Red float32 image of shape (M,N)
    green: np.ndarray
        Green float32 image of shape (M,N)
    blue: np.ndarray
        Blue float32 image of shape (M,N)
    perc: flaot
        Before conversion from float to integer types intensity is rescaled using percentiles, by default 1.

    Returns
    -------
    Tuple[np.ndarray,np.ndarray]
        rgb_ui8: np.ndarray
            RGBA uint8 image of shape (M,N,3)
        rgba_ui32: np.ndarray
            RGB uint32 image of shape (M,N)
    """
    shape = red.shape

    # RGB float32 image
    rgb = np.empty((shape[0], shape[1], 3), dtype=np.float32)
    rgb[:,:,0] = red
    rgb[:,:,1] = green
    rgb[:,:,2] = blue

    # Get NaN mask, i.e. positions where all rgb channels have NaN values
    mask = np.any(np.isnan(rgb), axis=2)

    # Get image percentiles
    perc = 1
    vmin = np.nanpercentile(rgb, perc)
    vmax = np.nanpercentile(rgb, 100-perc)

    # Convert image from np.float32 to np.uint8 by rescaling to percentiles
    rgb_ui8 = skimage.exposure.rescale_intensity(
        rgb,
        in_range=(vmin, vmax),
        out_range=(0,255),
    ).astype(np.uint8)

    # Convert RGB to RGBA image
    rgba_ui8 = cv2.cvtColor(rgb_ui8, cv2.COLOR_RGB2RGBA)

    # Convert RGBA (M,N,4) uint8 image to RGBA (M,N) uint32 image
    rgba_ui32 = rgba_ui8.view(np.uint32).reshape(shape[0], shape[1])

    # Apply mask to assign no color to NaN values
    rgb_ui8[mask] = 0
    rgba_ui32[mask] = 0
    
    return rgb_ui8, rgba_ui32