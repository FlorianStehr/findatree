from typing import List
from typing import Tuple
import numpy as np
import cv2 as cv

def blur(img, iterations=(0,5), kernel_size=3):
    img_blur = img.copy()

    for i in range(iterations[0]):
        img_blur = cv.medianBlur(img_blur,kernel_size)
    for i in range(iterations[1]):
        img_blur = cv.GaussianBlur(img_blur,(kernel_size,kernel_size),0)
    return img_blur


#%%
def _connectedComponents_idx(mask: np.ndarray) -> List[Tuple[np.ndarray]]:
    '''
    Return indices of connectedComponents (see openCV: connectedComponents) in a binary mask.

    Parameters:
    -----------
    mask: np.ndarray
        Binary mask where background -> 0 and foreground -> 1.
    
    Returns:
    --------
    List of ``len(connectedComponents)``. Each entry corresponds to np.ndarray containing (co) indices of respective connectedComponent in original mask.

    '''

    # Get connectedComponents in mask -> ccs: each cc is assigned with unique integer, background with 0
    count, ccs = cv.connectedComponents(mask.astype(np.uint8))
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


#%%
def divide_connectedComponents(mask, percentile=25):
    
    # Get indices of connectedComponents (ccs) in mask 
    ccs_idx = _connectedComponents_idx(mask)

    # Distance trafo of mask
    dist_trafo = cv.distanceTransform(mask.astype(np.uint8), cv.DIST_L2, 5)

    # Initiate new mask
    new_mask = dist_trafo.astype(np.float32)
    
    # Go through ccs and create sub-ccs
    for idx in ccs_idx:
        crit = np.percentile(new_mask[idx], percentile)
        isle_max = np.max(new_mask[idx])
        
        if isle_max > crit:
            new_mask[idx] = new_mask[idx] - crit

    new_mask[new_mask <= 0] = 0
    new_mask[new_mask > 0] = 1

    new_mask = new_mask.astype(np.uint8) 
    
    return new_mask