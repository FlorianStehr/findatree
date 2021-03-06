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
import skimage.feature as feature

import findatree.io as io

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
def local_thresholding(img_in, mask, width, resolution, blur=True):
    '''
    Local gaussian thresholding of image.
    '''
    # Set everything outside global mask to 0
    img = img_in.copy()
    img[np.invert(mask)] = 0

    # Apply median blurring before application of local threshold
    if blur:
        img = filters.median(
            img,
            footprint=morph.square(3),
            )

    # Kernel width
    width_px = width / resolution
    width_px = int(np.round(width_px))
    if width_px % 2 == 0:
        width_px +=1 
    
    # Print kernel width
    print(f"Gaussian thresholding kernel width: {width:.1f} [m] = {width_px:.0f} [px]")

    # Local thresholding
    thresh = filters.threshold_local(
        img,
        block_size=width_px,
        method='gaussian',
    )
    mask_local = (img > thresh).astype(np.uint8)

    return mask_local


#%%
def watershed_by_peaks_in_disttrafo(image, mask, peak_min_distance, resolution):
    '''
    Marker based watersheding of image.
    Markers are generated by:
        1. Removing small holes (``area = 0.5 [m**2]``) of mask
        2. Computing distance transform (-> ``dist``) of mask 
        3. Finding local peaks within ``dist`` separated by minimum distance ``min_distance``.
    Marker based watersheding is finally carried out on ``- dist * image``.
    '''

    # Remove small holes in mask
    area_thresh = 0.25
    area_thresh_px = int(np.round(area_thresh / resolution**2))
    print(f"Removing holes of area: {area_thresh:.2f} [m**2] = {area_thresh_px:.0f} [px]")
    mask = morph.remove_small_holes(
        mask.astype(np.bool_),
        area_threshold=area_thresh_px,
        connectivity=1,
    )
    mask = mask.astype(np.uint8)

    # Distance transform of mask
    dist = distance_transform(mask)

    # Set & print minimum distance in [m] and [px]
    peak_min_distance_px = int(np.round(peak_min_distance / resolution))
    print(f"Peaks in distance transform separated by minimum distance: {peak_min_distance:.1f} [m] = {peak_min_distance_px:.0f} [px]")

    # Find local peaks in distance transform of mask
    peaks_idx = feature.peak_local_max(dist, min_distance=peak_min_distance_px)
    peaks = np.zeros_like(mask,dtype=np.uint8)
    peaks[tuple(peaks_idx.T)] = 1

    # Dilate peaks a bit
    peaks = morph.dilation(
        peaks,
        footprint=morph.disk(3),   
    )

    # Prepare output mask + seed
    mask_seed = mask + peaks

 
    # Label peaks to generate markers
    markers = measure.label(peaks, background=0, return_num=False, connectivity=2)

    # Watershed
    labels = segmentation.watershed(
        - dist * image,
        markers=markers, 
        mask=mask,
        watershed_line=True,
    )

    # Remove contracted, small labels after watershed
    labels = morph.remove_small_objects(
        labels,
        min_size=5,
        connectivity=1,
    )

    # Relabel in sequential manner after removing contracted labels
    labels = segmentation.relabel_sequential(labels)[0]

    # Watershed again, after removing contracted and sequential relabeling
    labels = segmentation.watershed(
        - dist * image,
        markers=labels, 
        mask=mask,
        watershed_line=True,
    )

    # Boundaries of labels
    bounds = labels_to_bounds(labels)

    return labels, bounds, mask_seed


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

    # Expand final labels quite a bit (possible due to next step of restriction to initial mask)
    labels_final = segmentation.expand_labels(labels_final,distance=2)
    
    # Restrict labels to original mask
    labels_final[mask == 0] = 0
    
    # Get the final remainder between the mask and the final labels to labels the so far unlabeled part of the mask
    mask_remain = mask.copy()
    mask_remain[labels_final > 0] = 0

    # Remove very small objects from remainder mask
    mask_remain = morph.remove_small_objects(
        mask_remain.astype(np.bool_),
        min_size=4,
        connectivity=1,
    )
    mask_remain = mask_remain.astype(np.uint8)
    mask_seed = mask_seed + mask_remain

    # Label the final remainder mask
    labels_remain = measure.label(mask_remain, background=0, return_num=False, connectivity=2)
    labels_remain[labels_remain > 0] = labels_remain[labels_remain > 0] + np.max(labels_final.flatten())
    labels_final = labels_final + labels_remain
    
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


#%%
def main(
    cs_primary,
    resolution,
    downscale=1,
    thresh_channel='l',
    thresh_width=30,
    thresh_blur=True,
    water_channel='l',
    water_peakdist=1.2,
    ):
    
    print(f"Pixel width: {resolution:.1f} [m]")
    

    ######################################### (1) Normalize channels for thresholding and watershed
    cs_thresh, shape_thresh = io.define_channels(cs_primary, reduce=downscale)
    cs_water, shape_water = io.define_channels(cs_primary, reduce=0)

    # Print infos
    resolution_thresh = resolution * 2**downscale
    print("Thresholding:")
    print(f"   * Channel: {thresh_channel}")
    print(f"   * Pixel width: {resolution_thresh:.1f} [m]")
    print("Watershed:")
    print(f"   * Channel: {thresh_channel}")
    print(f"   * Pixel width: {resolution:.1f} [m]")

    
    ######################################### (2) Set global mask
    mask_global_thresh = (cs_thresh['chm'] > 3) & (cs_thresh['ndvi'] > 0.4)


    ######################################### (3) Local gaussian thresholding
    # Channel used for local thresholding
    img_thresh = cs_thresh[thresh_channel].copy()

    # Gaussian local thresholding
    mask_local_thresh = local_thresholding(
        img_thresh,
        mask=mask_global_thresh,
        width=thresh_width,
        resolution=resolution_thresh,
        blur=thresh_blur,
        )


    ######################################### (4) Re-upscale masks for watershed
    # Upscale tresholding masks to original resolution
    mask_global = labels_resize(mask_global_thresh, shape=shape_water).astype(np.bool_)
    mask_local_water = labels_resize(mask_local_thresh, shape=shape_water).astype(np.bool_)


    ######################################### (5) Marker based watershed based on local peaks in distance transform of local mask
    # Channel used for watershed
    img_water = cs_water[water_channel].copy()

    # Label by marker based watershed based on local peaks of distance transform of local mask
    labels, bounds, mask_seed = watershed_by_peaks_in_disttrafo(
        img_water,
        mask=mask_local_water,
        peak_min_distance=water_peakdist,
        resolution=resolution,
        )

    return labels, bounds, mask_seed, mask_global

