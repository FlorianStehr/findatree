from typing import List
from typing import Tuple
import numpy as np
import cv2 as cv
import skimage
import numba

#%%
def scaler_minmax_uint8(img_in, mask=None):
    
    # Copy input image
    img = img_in.copy()

    # Define mask if not set
    if mask is None:
        mask = np.ones(img.shape, dtype=np.bool_)

    # Min/Max scaling
    img -= np.nanmin(img[mask])
    img *= 255 / np.nanmax(img[mask])
    
    # dtype conversion
    img = img.astype(np.uint8)
    img[np.invert(mask)] = 0

    return img


#%%
def local_thresholding(img_in, mask_global, distance):

    # MinMax scaling and conversion to uint8 image
    img = scaler_minmax_uint8(img_in, mask_global)
    
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

    # Morphological trafos
    kernel = np.ones((3,3), dtype=np.uint8)
    # mask_local = cv.erode(mask_local, kernel)
    # mask_local = cv.dilate(mask_local, kernel)
    # mask_local = cv.dilate(mask_local, kernel)
    # mask_local = cv.erode(mask_local, kernel)
    

    return mask_local


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
