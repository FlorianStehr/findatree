from typing import Dict, List, Tuple
import numpy as np
import cv2 as cv
import numba
import importlib

from scipy.spatial import distance_matrix
from scipy.ndimage.morphology import distance_transform_edt as distance_transform

import skimage.measure as measure
import skimage.transform as transform
import skimage.filters as filters
import skimage.morphology as morph
import skimage.segmentation as segmentation
import skimage.feature as feature

import findatree.io as io

importlib.reload(io)

#%%
def local_thresholding(img_in, mask, width, px_width, blur=True):
    '''Local gaussian thresholding of image.
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
    width_px = width / px_width
    width_px = int(np.round(width_px))
    if width_px % 2 == 0:
        width_px +=1 
    
    # Print kernel width
    print(f"    ... [segmentation.local_thresholding()] Gaussian thresholding kernel width: {width:.1f} [m] = {width_px:.0f} [px]")

    # Local thresholding
    thresh = filters.threshold_local(
        img,
        block_size=width_px,
        method='gaussian',
    )
    mask_local = (img > thresh).astype(np.uint8)

    return mask_local


#%%
def watershed_by_peaks_in_disttrafo(
    image: np.ndarray,
    mask: np.ndarray,
    px_width: float,
    hole_min_area: float,
    peak_min_distance: float,
    label_min_area: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Marker based watershed of image within mask. Markers are generaated by local peaks in distance transform of mask.

    Parameters
    ----------
    image : np.ndarray
        Channel that is multiplied with distance transform of mask to perform watershed.
    mask : np.ndarray
        Local mask as obtained by `segmentation.local_thresholding()` as np.ndarray of type np.uint8.
    px_width : float
        Pixel width of image/mask in meters.
    hole_min_area : float
        Holes smaller than this area in meters will be removed from local mask prior to thresholding.
    peak_min_distance : float
        Min. distance between peaks of `distance transform * image` for marker generation.
    label_min_area : float
        Labels smaller than this area in meters will be removed, than relabeling and re-waterhed using labels as markers.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        labels: np.ndarray
            Labels of same size as image as of type np.int64. Every label corresponds to one unique (continuous) integer.
        mask_seed: np.ndarray
            Mask and seed/marker image. Background=0, Mask=1, Seed/Marker=2.

    Notes:
    ------
    1. Remove small holes of `mask` which are `< area_threshold`
    2. Computing distance transform (-> `dist`) of `mask`
    3. Finding local peaks (-> `peaks`) within `dist` separated by minimum distance `peak_min_distance`.
    4. Marker based watershed using `peaks` is carried out on `- dist * image` within mask (-> `labels`)
    5. Small labels are removed which are `< label_min_size`
    6. Relabeling for continuous indices of labels
    7. Repeat step 5 with labels after step 6 as markers

    """

    # Remove small holes in mask
    hole_min_area_px = int(np.round(hole_min_area / px_width**2))
    print(f"    ... [segmentation.watershed_by_peaks_in_disttrafo()] Removing holes of area: {hole_min_area:.2f} [m**2] = {hole_min_area_px:.0f} [px]")
    mask = morph.remove_small_holes(
        mask.astype(np.bool_),
        area_threshold=hole_min_area_px,
        connectivity=1,
    )
    mask = mask.astype(np.uint8)

    # Distance transform of mask
    dist = distance_transform(mask)

    # Set minimum distance in [m] and [px]
    peak_min_distance_px = int(np.round(peak_min_distance / px_width))
    print(f"    ... [segmentation.watershed_by_peaks_in_disttrafo()] Local peaks min. distance: {peak_min_distance:.2f} [m] = {peak_min_distance_px:.0f} [px]")

    # Find local peaks in distance transform of mask
    peaks_idx = feature.peak_local_max(dist, min_distance=peak_min_distance_px)
    peaks = np.zeros_like(mask,dtype=np.uint8)
    peaks[tuple(peaks_idx.T)] = 1
 
    # Label peaks to generate markers
    markers = measure.label(peaks, background=0, return_num=False, connectivity=2)

    # Expand marker without overlap before watershed
    marker_expansion = int(np.floor(0.8 / px_width))
    markers = segmentation.expand_labels(
        markers,
        distance=marker_expansion,
    )
    markers[np.invert(mask.astype(np.bool_))] = 0 # Restrict markers to mask

    # Prepare output mask + markers -> mask_seed
    mask_seed = mask + markers
    mask_seed[mask_seed > 2] = 2

    # Watershed
    labels = segmentation.watershed(
        - dist * image,
        markers=markers, 
        mask=mask,
        watershed_line=True,
    )

    # Remove small labels after watershed
    label_min_area_px = int(np.round(label_min_area / px_width**2))
    print(f"    ... [segmentation.watershed_by_peaks_in_disttrafo()] Removing labels of area: {label_min_area:.2f} [m**2] = {label_min_area_px:.0f} [px]")
    labels = morph.remove_small_objects(
        labels,
        min_size=label_min_area_px,
        connectivity=1,
    )

    # Relabel in sequential manner after removing small labels
    labels = segmentation.relabel_sequential(labels)[0]

    # Watershed again, after removing contracted and sequential relabeling
    labels = segmentation.watershed(
        - dist * image,
        markers=labels, 
        mask=mask,
    )

    return labels, mask_seed


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
    cs: Dict,
    params_cs: Dict,
    params: Dict,
    verbose: bool=True,
    ) -> Tuple[Dict,Dict]:
    """Local gaussian thresholding and marker based watershed on image to segment crowns.

    Parameters
    ----------
    cs : Dict[np.ndarray,]
        Dictionary of of secondary channels as np.ndarray of dtype=np.float64, see io.channels_primary_to_secondary().
    params_cs : Dict
         Dictionary of parameters of secondary channels, see io.channels_primary_to_secondary().
    params : Dict
        Parameters used during function call with following keys:
        * thresh_global_chm [float]: Global chm threshold, by default 3.
        * thresh_global_ndvi [float]: Global ndvi threshold, by default 0.4.
        * thresh_channel: Local thresholding channel, by default l. 
        * thresh_downscale [int]: Downscale factor before local thresholding, by default 1,
        * thresh_blur: Apply median blur with kernel size 3 before local thresholding, by default False.
        * thresh_width [float]: Local thresholding kernel size in meters, by default 30.
        * water_channel: If `None` watershed is carried out on distance transform of local mask only, otherwise distance transfrom is multiplied with respective channel, by default l.
        * water_downscale: Downscale factor before watershed, by default 0.
        * water_hole_min_area [float]: Remove holes smaller than this threshold in meters^2 from local mask before generating markers by use of distance transform, by default 0..
        * water_peak_dist [float]: Minimum peak distance in meters between local maxima of distance transform of local mask, by default 1.2.
        * water_label_min_area [float]: Labels smaller than this value in meters^2 are removed and relabeling is performed, by default 0.2.
    verbose : bool, optional
        Print parameters during call, by default True.

    Returns
    -------
    Tuple[Dict,Dict]
        cs_segment: dict[np.ndarray,...]
            Dictionary of labels as np.ndarray of dtype=np.float64 with same shape as images in original `cs`.
            Keys are: ['labels', 'bound', 'blue', 'green', 'red', 're', 'nir'].
        params: dict
            Parameters used during function call.

    Notes:
    ------
    1. Global mask is generated by selection `(chm > thresh_global_chm) & (ndvi > thresh_global_ndvi)`.
    2. Local gaussian thresholding is applied on channel `thresh_channel` with kernel size `thresh_width`. Optinonally blur is applied beforehand with `thresh_blur`.
    3. Markers are generated by finding local peaks (see `water_peak_dist`) within distance transform `dist` of local mask (see `water_hole_min_area`).
    4. Watershed is performed on `dist * water_channel` using locak peaks as markers.
    5. Final removal of small labels (see `water_label_min_area`) and relabeling.

    Resolution for thresholding and watershed can be adjusted using `thresh_downscale` and `water_downscale`.
    However, final labels will be of same size channels in `cs`.

    See: `segmentation.local_thresholding()`, `segmentation.watershed_by_peaks_in_disttrafo`, `io.channels_primary_to_secondary`.
        
    """

    ######################################### (0) Set standard settings if not set
    params_standard = {
        'thresh_global_chm': 3,
        'thresh_global_ndvi': 0.4,
        'thresh_channel': 'l',
        'thresh_downscale': 1,
        'thresh_blur': False,
        'thresh_width': 30,
        'water_channel': 'l',
        'water_downscale': 0,
        'water_peak_dist': 1.2,
        'water_hole_min_area': 0.,
        'water_label_min_area': 0.2,
    }
    params_keys = [k for k in params]
    for k in params_standard:
        if k in params:
            params[k] = params[k]
        else:
            params[k] = params_standard[k]

    ######################################### (1) Prepare channels at defined px_widths for thresholding and watershed
    cs_thresh, params_thresh = io.channels_primary_to_secondary(cs, params_cs, downscale=params['thresh_downscale'], verbose=False)
    params['thresh_shape'] = params_thresh['shape']
    params['thresh_px_width'] = params_thresh['px_width']

    cs_water, params_water = io.channels_primary_to_secondary(cs, params_cs, downscale=params['water_downscale'], verbose=False)
    params['water_shape'] = params_water['shape']
    params['water_px_width'] = params_water['px_width']


    ######################################### (2) Set global mask
    mask_global_thresh = (cs_thresh['chm'] > params['thresh_global_chm']) & (cs_thresh['ndvi'] > params['thresh_global_ndvi'])


    ######################################### (3) Local gaussian thresholding
    # Channel used for local thresholding
    img_thresh = cs_thresh[params['thresh_channel']].copy()

    # Gaussian local thresholding
    mask_local_thresh = local_thresholding(
        img_thresh,
        mask=mask_global_thresh,
        width=params['thresh_width'],
        px_width=params['thresh_px_width'],
        blur=params['thresh_blur'],
        )


    ######################################### (4) Re-upscale masks for watershed
    # Upscale tresholding masks to original resolution
    mask_global_water = labels_resize(mask_global_thresh, shape=params['water_shape']).astype(np.bool_)
    mask_local_water = labels_resize(mask_local_thresh, shape=params['water_shape']).astype(np.bool_)


    ######################################### (5) Marker based watershed based on local peaks in distance transform of local mask
    # Channel used for watershed
    if params['water_channel'] is None:
        img_water = np.ones_like(mask_local_water)
    else:
        img_water = cs_water[params['water_channel']].copy()

    # Label by marker based watershed based on local peaks of distance transform of local mask
    labels_water, mask_seed_water = watershed_by_peaks_in_disttrafo(
        img_water,
        mask=mask_local_water,
        peak_min_distance=params['water_peak_dist'],
        px_width=params['water_px_width'],
        hole_min_area=params['water_hole_min_area'],
        label_min_area=params['water_label_min_area'],
    )

    ######################################### (6) Resize labels/bounds/masks
    labels = labels_resize(labels_water, shape=params_cs['shape'])
    mask_global = labels_resize(mask_global_water, shape=params_cs['shape'])
    mask_seed = labels_resize(mask_seed_water, shape=params_cs['shape'])
    bounds= labels_to_bounds(labels)

    # Prepare return channels
    cs_segment = {}
    cs_segment['labels'] = labels
    cs_segment['bounds'] = bounds
    cs_segment['mask_seed'] = mask_seed
    cs_segment['mask_global'] = mask_global

    # Print parameters
    if verbose:
        print('-----------')
        print('Parameters:')
        for k in params: print(f"  {k:<30}: {params[k]}")

    return cs_segment, params

