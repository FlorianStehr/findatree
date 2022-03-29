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
def _connectedComponents_idx(mask_in: np.ndarray) -> List[Tuple[np.ndarray]]:
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
    count, ccs = cv.connectedComponents(mask)
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
def _gradient(img) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Returns image gradient magnitude and direction.
    '''
    shape = img.shape
    img = img.astype(np.float32) 
    
    sobelx = cv.Sobel(img,cv.CV_32F,1,0,ksize=-1)
    sobely = cv.Sobel(img,cv.CV_32F,0,1,ksize=-1) 

    # grad_mag = np.hypot(sobelx, sobely) / 32          # Divide by the sum  of the Scharr weights to get closer to the real gradient magnitude
    grad_mag = (np.abs(sobelx) + np.abs(sobely)) / 32   # Divide by the sum  of the Scharr weights to get closer to the real gradient magnitude

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
 
            if (315 < direction) or (direction <= 45) or (135 < direction <= 225): 
                before_pixel = grad_mag[row, col - 1]
                after_pixel = grad_mag[row, col + 1]
            
            elif (45 < direction <= 135) or (225 < direction <= 315):
                before_pixel = grad_mag[row - 1, col]
                after_pixel = grad_mag[row + 1, col]
 
            if grad_mag[row, col] >= before_pixel and grad_mag[row, col] >= after_pixel:
                output[row, col] = grad_mag[row, col]
 
    return output

#%%
def _hysteresis(img, thresh_lower, thresh_upper, thresh_len):
    
    # Prepare mask
    mask = img.copy()
    mask[mask < thresh_lower] = 0 # Remove lower threshold values
    mask[mask > 0] = 1
    mask = mask.astype(np.uint8)

    # Get indices of connectedComponents (ccs) in mask 
    ccs_idx = _connectedComponents_idx(mask)
    
    # Loop through connectedComponents and keep those that fullfill upper threshold condition
    new_mask=  np.zeros(img.shape, dtype=np.uint8)
    for i, idx in enumerate(ccs_idx):
        cond1 = len(idx[0]) >= thresh_len                   # Defintion for length condition
        cond2 = np.sum(img[idx] > thresh_upper) >= 2        # Defintion for upper threshold condition
        cond = cond1 & cond2
        if cond:
            new_mask[idx] = 1

    return new_mask



#%%
def _canny_edge(img_in: np.ndarray, mask: np.ndarray, kernel: int=3):
    
    img = img_in.copy()
    
    # Blur image
    if kernel > 0:
        for i in range(10):
            img = cv.medianBlur(img, kernel)
            img = cv.GaussianBlur(img, (kernel, kernel), 0)

    # Compute gradient magitude and direction
    grad_mag, grad_dir = _gradient(img)

    # Normalize gradient to median value of iamge within mask
    grad_mag_scaler = np.nanpercentile(img[mask], 95) - np.nanpercentile(img[mask], 5)
    grad_mag = grad_mag / grad_mag_scaler

    # Set all gradient magnitudes outside of mask to zero
    grad_mag[np.invert(mask)] = 0

    # Thining based on non-max suppression
    edge = _non_max_suppression(grad_mag, grad_dir)

    # Get maximum of egde-gradient-magnitude distribution (used later for determining hyteresis thresholds!)
    edge_vals = edge[mask]
    edge_vals = edge_vals[edge_vals > 0]
    vals, bins = np.histogram(edge_vals, bins=np.linspace(0, 0.5, 100))
    thresh = bins[np.argmax(vals)]

    # Hysteresis
    thresh_min = np.percentile(edge[edge > thresh], 0)
    thresh_max = np.percentile(edge[edge > thresh], 50)
    thresh_len = 10
    edge = _hysteresis(edge, thresh_min, thresh_max, thresh_len)

    return img, grad_mag, edge


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
def seeds_by_gradient(channels: List[np.ndarray,], mask: np.ndarray):
    
    shape = (channels[0].shape[0], channels[0].shape[1], len(channels))
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))

    # Init
    img = np.zeros(shape, dtype=np.float32)
    grad = np.zeros(shape, dtype=np.float32)
    edge = np.zeros(shape, dtype=np.uint8)

    # Canny edge detection for each channel
    for i, c in enumerate(channels):
        img[:,:,i], grad[:,:,i], edge[:,:,i] = _canny_edge(c, mask)

    # Sum edges
    edge_sum = np.sum(edge, axis=-1)
    edge_sum[edge_sum > 0] = 1
    edge_sum[edge_sum <= 0] = 0
    edge_sum = edge_sum.astype(np.uint8)

    # Morphological closing of summed edges
    edge_sum = cv.dilate(edge_sum, kernel, iterations=1)
    edge_sum = cv.erode(edge_sum, kernel, iterations=1)

    # Combined maximum gradient image
    grad_max = np.max(grad, axis=-1)

    # Seed for marker generation by shrinking
    seed = mask.astype(np.uint8)
    seed[edge_sum == 1] = 0
    
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    seed = cv.erode(seed, kernel, iterations=1)

    # Generate markers by seed shrinking
    markers = seed.copy()
    for i in range(10):
        markers = divide_connectedComponents(markers, percentile=50)

    # Generate marker_combo for watershed
    # Aim is to set:
    #   sure_bg -> 1
    #   unknown -> 0
    #   marker  -> >=2
    marker_combo = np.invert(mask).astype(np.int16) # Now: sure_bg -> 1
    count, markers = cv.connectedComponents(markers, connectivity=8, ltype=cv.CV_16U)
    markers = markers + 1
    markers[markers == 1] = 0 # Now all outside markers -> 0, markers -> >=2
    marker_combo = marker_combo + markers
    # marker_combo[edge_sum == 1] = 1

    # Prepare grad_max as image for watershed
    grad_max_int8 = grad_max.copy()
    grad_max_int8[np.invert(mask)] = 0
    grad_max_int8 -= np.min(grad_max_int8[mask])
    grad_max_int8 *= (255 / np.max(grad_max_int8[mask]))

    img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    for i in range(3):
        img[:,:,i] = grad_max_int8

    # Watershed
    ws_out = cv.watershed(img, marker_combo.copy())

    marker_combo[edge_sum == 1] = 1
    
    return grad_max, edge_sum, marker_combo, ws_out
    
    
    

