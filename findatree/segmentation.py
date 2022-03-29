from typing import List
from typing import Tuple
import numpy as np
import cv2 as cv
import numba

from scipy.spatial import distance_matrix
from scipy.ndimage.morphology import distance_transform_edt as distance_transform

import skimage.measure as measure
import skimage.transform as transform
import skimage.filters as filters
import skimage.morphology as morph
import skimage.segmentation as segmentation

#%%
def scaler_percentile(img_in, percentile=0, mask=None, return_dtype='uint8'):
    '''
    Rescale image to interval ``[0,1]`` based on percentile ``(p_lower = percentile, p_upper = 100 - percentile)``
    '''

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
    '''
    Local gaussian thresholding of image.
    '''

    # Set everything outside global mask to 0
    img = img_in.copy()
    img[np.invert(mask_global)] = 0

    # Kernel size
    distance = int(np.round(distance))
    if distance % 2 == 0:
        distance +=1 
    print(f"Distance set to: {distance} [px]")

    # Local thresholding
    thresh = filters.threshold_local(
        img,
        block_size=distance,
        method='gaussian',
    )
    mask_local = (img > thresh).astype(np.uint8)

    return mask_local


#%%
def shrinkmask_expandlabels(mask_in, thresh_dist=2, verbose=False):
    '''
    Shrink local mask by distance threshold and then re-expand resulting labels by approx. the same distance.
    Return labels and remainder mask between labels and orginal local mask for next iteration 
    '''

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
        area_threshold=2,
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
    mask_shrink = mask_shrink.astype(np.uint8)
    
    # Expand labels by thresh_dist and compute overlap with mask
    labels = measure.label(mask_shrink, background=0, return_num=False, connectivity=2)
    labels = segmentation.expand_labels(labels,distance=thresh_dist * 1.2)
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

def shrinkmask_expandlabels_iter(mask_in, thresh_dist_start, thresh_dist_stop=1, verbose=False):
    '''
    Iterative shrinking of local mask and expansion of remaining labels for decreasing distance thresholds.
    '''
    # Define initial values
    shape = mask_in.shape
    mask_remain = mask_in.copy()
    thresh_dist = thresh_dist_start

    labels_masks_it= []
    threshs_it = []
    labels_final = np.zeros(shape, dtype=np.uint32)
    mask_seed = np.zeros(shape, dtype=np.uint8)

    while thresh_dist > thresh_dist_stop:
        # Shrink mask & expand labels
        labels_masks, threshs = shrinkmask_expandlabels(mask_remain, thresh_dist=thresh_dist, verbose=verbose)
        
        # Assign iteration results to lists
        labels_masks_it.append(labels_masks)
        threshs_it.append(threshs)

        # Assign values for next iteration
        thresh_dist = thresh_dist - 1
        mask_remain = labels_masks[1]

        # Add up labels
        labels = labels_masks[0]
        labels[labels > 0] = labels[labels > 0] + np.max(labels_final.flatten())
        labels_final = labels_final + labels

        # Add current seeds
        mask_seed += labels_masks[3]
    

    # Define return mask and mask_seed
    mask = labels_masks_it[0][2] # Initial mask with small holes and objects removed
    mask_seed = mask_seed + mask

    # Expand final labels quite a bit (possible due to next step of restriction to inital dilated mask)
    labels_final = segmentation.expand_labels(labels_final,distance=3)
    
    # Resrict expanded labels to dilated original mask
    mask_dilated = morph.dilation(
        mask,
        footprint=morph.disk(1),
    ).astype(np.uint8)
    labels_final[mask_dilated == 0] = 0

    # Boundaries
    bounds = labels_to_bounds(labels_final)

    return labels_final, bounds, mask_seed


#%%
def labels_to_bounds(labels):
    '''
    Return boundaries of labels, i.e. integer labeled connected components in image.
    '''
    bounds = segmentation.find_boundaries(
        labels,
        connectivity=1,
        mode='outer',
    ).astype(np.uint8)
    bounds = morph.thin(bounds)

    return bounds


#%%
def labels_resize(labels, shape):
    '''
    Resize labels to a defined shape.
    '''
    labels_resize = transform.resize(
        labels,
        output_shape=shape,
        order=0,
        preserve_range=True,
        )
    
    return labels_resize

