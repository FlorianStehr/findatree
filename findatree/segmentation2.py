from typing import List
from typing import Tuple
import numpy as np
import cv2 as cv
import numba

from scipy.spatial import distance_matrix
from scipy.ndimage.morphology import distance_transform_edt as distance_transform

import skimage.measure as measure
import skimage.morphology as morph
import skimage.segmentation as segmentation

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
def shrinkmask_expandlabels(mask_in, thresh_dist=2, verbose=False):
    
    mask = mask_in.copy()

    # Set area threshold for removing small objects/holes
    thresh_area = max(np.ceil(thresh_dist**2 / 2), 1)
    if verbose:
        print()
        print(f"Distance threshold: {thresh_dist:.1f} px")
        print(f"Area threshold:     {thresh_area:.0f}  px")

    # Remove small holes
    mask = morph.remove_small_holes(
        mask.astype(np.bool_),
        area_threshold=4,
        connectivity=1,
    )

    # Remove small objects
    mask = morph.remove_small_objects(
        mask,
        min_size=4,
        connectivity=1,
    )
    mask = mask.astype(np.uint8)

    # Distance transform
    dist = distance_transform(mask)

    # Threshold distance transform by thresh_dist -> mask_shrink
    mask_shrink = dist.copy()
    mask_shrink[dist < thresh_dist] = 0
    mask_shrink[dist >= thresh_dist] = 1
    
    # Expand labels by thresh_dist and compute overlap with mask
    labels = measure.label(mask_shrink, background=0, return_num=False, connectivity=2)
    labels = segmentation.expand_labels(labels,distance=thresh_dist * np.sqrt(2))
    labels = labels * mask

    # Remove very small labeled objects
    mask_labels_large = labels.copy()
    mask_labels_large[labels > 0] = 1
    mask_labels_large[labels == 0] = 0
    mask_labels_large = morph.remove_small_objects(
        mask_labels_large.astype(np.bool_),
        min_size=thresh_area,
        connectivity=1,
    )
    mask_labels_large = mask_labels_large.astype(np.uint8)
    labels = labels * mask_labels_large
    
    # Now create remainder mask, i.e. objects that have not been labeled
    mask_remain = mask.copy()
    mask_remain[labels > 0 ] = 0

    # Define return values
    labels_masks = [labels, mask_remain, mask, mask_shrink]
    threshs = [thresh_dist, thresh_area]

    return labels_masks, threshs

#%%

def shrinkmask_expandlabels_iter(mask_in, thresh_dist_init, thresh_dist_lower=1, final_expansion=np.sqrt(2), verbose=False):
    
    # Define initial values
    shape = mask_in.shape
    mask_remain = mask_in.copy()
    thresh_dist = thresh_dist_init
    thresh_area = np.ceil(thresh_dist**2 / 2)

    labels_masks_it= []
    threshs_it = []
    labels_final = np.zeros(shape, dtype=np.uint32)
    mask_seed = np.zeros(shape, dtype=np.uint8)

    while thresh_dist > thresh_dist_lower:
        # Shrink mask & expand labels
        labels_masks, threshs = shrinkmask_expandlabels(mask_remain, thresh_dist=thresh_dist, verbose=verbose)
        
        # Assign iteration results to lists
        labels_masks_it.extend(labels_masks)
        threshs_it.extend(threshs)

        # Assign values for next iteration
        thresh_dist = thresh_dist / np.sqrt(2)
        mask_remain = labels_masks[1]

        # Add up labels
        labels = labels_masks[0]
        labels[labels > 0] = labels[labels > 0] + np.max(labels_final.flatten())
        labels_final = labels_final + labels

        # Add up mask & mask_shrink -> mask_seed
        mask = labels_masks[2]
        mask_shrink = labels_masks[3] 
        mask_seed = mask_seed + (mask - mask_remain) + mask_shrink
    
    # Expand final labels 
    labels_final = segmentation.expand_labels(labels_final,distance=final_expansion)

    # Boundaries
    bounds = segmentation.find_boundaries(
        labels_final,
        connectivity=1,
        mode='outer',
    ).astype(np.uint8)
    bounds = morph.thin(bounds)

    return labels_final, bounds, mask_seed, labels_masks_it


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
            # img = channels_in[key]
            channels[key] = img
            print(f"Channel: '{key}' from {thresh[0]:.1e} to {thresh[1]:.1e} mapped to [0,1]")


    return channels


#%%
def connectedComponents_idx(mask_in: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
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
    ccs = measure.label(mask, background=0, return_num=False, connectivity=1)

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


#%%
def connectedComponent_idx_tobox(idx: Tuple[np.ndarray, np.ndarray]) -> Tuple[Tuple, ...]:
    '''
    Project indices of a single connectedComponent to indices in the minimum spanning box. 
    '''
    box_origin = (np.min(idx[0]), np.min(idx[1])) 
    
    box_width = (
        np.max(idx[0]) - np.min(idx[0]) + 1,
        np.max(idx[1]) - np.min(idx[1]) + 1,
    )

    box_idx = (
        idx[0].copy() - box_origin[0],
        idx[1].copy() - box_origin[1],
    )
    return box_idx, box_origin, box_width


#%%
def connectedComponent_img_tobox(idx, img):
    '''
    Fill connectedComponent in minimum spanning box with data from image.
    '''
    box_idx, box_origin, box_width = connectedComponent_idx_tobox(idx)
    
    box = np.zeros((box_width[0], box_width[1]), dtype=np.float32)
    box[box_idx] = img[idx]

    return box


#%%
def connectedComponent_toarray(idx, cs):
    '''
    Create array of samples of a single connectedComponent. 
    One sample corresponds to a pixel of the connectedComponent.
    Features are the spatial coordinates xyz and three colors.
    '''
    # Init
    length = len(idx[0])
    data = np.zeros((length, 6), dtype=np.float32) # 3x spatial coordinates + 3x colors
    
    # Spatial coordinates
    data[:, 0] = cs['x'][idx]
    data[:, 1] = cs['y'][idx]
    data[:, 2] = cs['z'][idx]

    # Color channels
    data[:, 3] = cs['l'][idx]
    data[:, 4] = cs['ndvi'][idx]
    data[:, 5] = cs['s'][idx]
    

    return data

#%%
def connectedComponent_todistance(idx, cs, w=1):
    '''
    Return pairwise weigthed spatial/color distance between samples of a single connectedComponent.
    '''
    data = connectedComponent_toarray(idx, cs)
    dist_space = distance_matrix(data[:, :2], data[:, :2], p=2)
    dist_color = distance_matrix(data[:, 3:], data[:, 3:], p=2)

    dist = np.sqrt(dist_space**2 + w * dist_color**2)

    return data, dist


#%%
def nearestNeighbors_ball_inbox(idx, dist, r=10, i=None):

    if i is None:
        i = np.random.randint(0, len(idx[0]))
        print(i)

    box_idx = connectedComponent_idx_tobox(idx)[0]

    point = (box_idx[1][i], box_idx[0][i])
    
    nns_i = np.where(dist[i, :] <= r)[0]
    nns = np.zeros((len(nns_i), 3),)
    nns[:, 0] = box_idx[1][nns_i]
    nns[:, 1] = box_idx[0][nns_i]
    nns[:, 2] = dist[i, :][nns_i]

    return point , nns