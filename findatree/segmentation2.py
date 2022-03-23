from typing import List
from typing import Tuple
import numpy as np
import cv2 as cv
import skimage
import numba

#%%
def scaler_percentile(img_in, percentile=0, mask=None, return_dtype='uint8'):
    
    # Copy input image
    img = img_in.copy()

    # Define mask if not set
    if mask is None:
        mask = np.ones(img.shape, dtype=np.bool_)

    # Percentile scaling
    thresh_lower = max(np.nanpercentile(img[mask], percentile), 0)
    thresh_upper = np.nanpercentile(img[mask], 100 - percentile)
    img -= thresh_lower
    img /= (thresh_upper - thresh_lower)
    img[img < 0] = 0
    img[img > 1] = 1
    
    # dtype conversion
    if return_dtype == 'uint8': 
        img = img * 255
        img = img.astype(np.uint8)
        img[np.invert(mask)] = 0

    return img, (thresh_lower, thresh_upper)


#%%
def local_thresholding(img_in, mask_global, distance):

    # MinMax scaling and conversion to uint8 image
    img, thresh = scaler_percentile(
        img_in,
        percentile=0,
        mask=mask_global,
        return_dtype='uint8',
    )
    
    # Kernel size
    distance = int(np.round(distance))
    if distance % 2 == 0:
        distance +=1 
    print(f"Distance set to: {distance} [px]")

    # Gaussian adaptive thresholding
    mask_local = cv.adaptiveThreshold(
        img,                            # Source image 
        1,                              # Value assigned to pxs where condition is satisfied
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,  # Adaptive thresholding method
        cv.THRESH_BINARY,               # Thresholding type     
        distance,                       # Size of pixel neighborhood to calculate threshold
        0,                              # Constat substracted from the (weighted) mean prior to thresholding
    )

    return mask_local


#%%
def normalize_channels(channels_in, res_xy, res_z, mask=None, percentile=0):
    
    # Init
    channels = {}
    shape = channels_in[list(channels_in.keys())[0]].shape[:2]

    # Define mask if not set
    if mask is None:
        mask = np.ones(shape, dtype=np.bool_)
    # Convert mask to bool dtype
    mask = mask.astype(np.bool_)

    # Prepare x-y coordinates
    x = np.arange(0,shape[1],1,dtype=np.float32) * res_xy
    y = np.arange(0,shape[0],1,dtype=np.float32) * res_xy
    xx, yy = np.meshgrid(x,y)

    # Prepare z-coordinate
    z = channels_in['chm'].copy() * res_z

    # Add x-y-z-coordinates
    channels['x'] = xx 
    channels['y'] = yy 
    channels['z'] = z

    # Normalize color channels
    for key in channels_in:
        if key != 'chm':
            img, thresh = scaler_percentile(
                channels_in[key],
                percentile=percentile,
                mask=mask,
                return_dtype=None,
            )
            channels[key] = img
            print(f"Channel: '{key}' from {thresh[0]:.1e} to {thresh[1]:.1e} mapped to [0,1]")


    return channels


#%%
def connectedComponents_idx(mask_in: np.ndarray) -> List[Tuple[np.ndarray]]:
    '''
    Return indices of connectedComponents (see openCV: connectedComponents) in a binary mask.

    Parameters:
    -----------
    mask_in: np.ndarray
        Binary mask where background -> 0 and foreground -> 1.
    
    Returns:
    --------
    List of ``len(connectedComponents)``. Each entry corresponds to ``Tuple[np.ndarray,np.ndarray]`` containing (coordinate) indices of respective connectedComponent in original mask.

    '''

    # Get connectedComponents in mask -> ccs: each cc is assigned with unique integer, background with 0
    mask = mask_in.copy()
    mask = mask.astype(np.uint8)
    ccs = skimage.measure.label(mask, background=0, return_num=False, connectivity=1)

    # count, ccs = cv.connectedComponents(mask)
    ccs = ccs.flatten() # Flatten

    # Get number of unique counts in ccs plus the sort-index of ccs to reconstruct indices of each cc in mask
    ccs_len = np.unique(ccs, return_counts=True)[1]
    ccs_sidx = np.argsort(ccs)

    # Now loop through each cc and get indices in mask
    ccs_midx = []
    i0 = 0
    for l in ccs_len:
        i1 = i0 + l
        cc_midx = ccs_sidx[i0:i1]
        i0 = i1

        # Go from flattened index to coordinate index in original mask
        cc_midx = np.unravel_index(cc_midx, mask.shape) 
        
        ccs_midx.append(cc_midx)

    ccs_midx = ccs_midx[1:] # Remove index belonging to background (value of zero in cv.connectedComponents)

    return ccs_midx
