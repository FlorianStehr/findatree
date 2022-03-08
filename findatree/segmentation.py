from typing import List
from typing import Tuple
import numpy as np
import cv2 as cv
import numba

#%%
def blur(img, iterations=3, kernel_size=5):
   
    img_blur = img.copy()
    for i in range(iterations):
        img_blur = cv.medianBlur(img_blur, kernel_size)

    for i in range(iterations):
        img_blur = cv.GaussianBlur(img_blur, (kernel_size, kernel_size), 0)

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
    List of ``len(connectedComponents)``. Each entry corresponds to ``Tuple[np.ndarray,np.ndarray]`` containing (coordinate) indices of respective connectedComponent in original mask.

    '''

    # Get connectedComponents in mask -> ccs: each cc is assigned with unique integer, background with 0
    count, ccs = cv.connectedComponents(mask.astype(np.uint8), 8)
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


#%%
def _gradient(img) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Returns image gradient magnitude and direction.
    '''
    img = img.astype(np.float32) 
    
    sobelx = cv.Sobel(img,cv.CV_32F,1,0,ksize=-1)
    sobely = cv.Sobel(img,cv.CV_32F,0,1,ksize=-1) 

    grad_mag = np.hypot(sobelx, sobely) / 32

    grad_dir = np.arctan2(sobely, sobelx)
    grad_dir = np.rad2deg(grad_dir)
    grad_dir += 180

    return grad_mag, grad_dir


#%%

@numba.jit(nopython=True, nogil=True, cache=False)
def _non_max_suppression(grad_mag: np.ndarray, grad_dir: np.ndarray) -> np.ndarray:
    shape = grad_mag.shape
    
    output = np.zeros(shape, dtype=grad_mag.dtype)
    
    for row in range(1, shape[0] - 1):
        for col in range(1, shape[1] - 1):
            direction = grad_dir[row, col]
 
            if (300 < direction) or (direction <= 60) or (120 < direction <= 210):
                before_pixel = grad_mag[row, col - 1]
                after_pixel = grad_mag[row, col + 1]
            
            elif (60 < direction <= 120) or (210 < direction <= 300):
                before_pixel = grad_mag[row - 1, col]
                after_pixel = grad_mag[row + 1, col]
 
            if grad_mag[row, col] >= before_pixel and grad_mag[row, col] >= after_pixel:
                output[row, col] = grad_mag[row, col]
 
    return output


#%%
def _hysteresis(img, thresh_min, thresh_max, thresh_len):
    
    mask = img.copy()
    mask[mask < thresh_min] = 0 # Remove lower threshold values
    mask[mask > 0] = 1
    mask = mask.astype(np.uint8)

    # Get indices of connectedComponents (ccs) in mask 
    ccs_idx = _connectedComponents_idx(mask)

    new_img =  np.zeros(img.shape, dtype=np.float32)
    
    for i, idx in enumerate(ccs_idx):

        if (len(idx[0]) > thresh_len) & (np.sum(img[idx] > thresh_max) >= 1):
            new_img[idx] = 1
    
    return new_img



#%%
def _canny_edge(img_in: np.ndarray, mask: np.ndarray):
    
    img = img_in.copy()
    shape = img.shape
    
    # Blur image
    kernel_size = 5
    for i in range(10):
        img = cv.medianBlur(img, kernel_size)
        img = cv.GaussianBlur(img, (7, 7), 0)
    
    # Flatten lower and upper values in image
    img_min = np.nanpercentile(img[mask],10)
    img_max = np.nanpercentile(img[mask],90)
    # img[img < img_min] = img_min
    # img[img > img_max] = img_max

    # Compute gradient magitude and direction
    grad_mag, grad_dir = _gradient(img)

    # Normalize gradient to median value of iamge within mask
    grad_mag_scaler = np.nanpercentile(img[mask],95) - np.nanpercentile(img[mask],5)
    grad_mag = grad_mag / grad_mag_scaler

    grad_mag[np.invert(mask)] = 0

    # Thining based on non-max suppression
    edges = _non_max_suppression(grad_mag, grad_dir)

    # Hysteresis
    thresh_min = np.percentile(edges[edges > 0], 25)
    thresh_max = np.percentile(edges[edges > 0], 50)
    thresh_len = 10
    edges = _hysteresis(edges, thresh_min, thresh_max, thresh_len)

    return img, edges

    



