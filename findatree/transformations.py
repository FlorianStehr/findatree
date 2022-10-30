from typing import Dict, List, Tuple
import time
import numpy as np
import cv2

import skimage.exposure
import skimage.segmentation
import skimage.morphology
import skimage.feature
import cv2

import rasterio
import rasterio.features

import shapely.geometry


#%%
def current_datetime() -> str:
    """Create a string of the current date and time.

    Returns
    -------
    str
        Current date and time in format `'%Y%m%d-%H%M%S'`
    """
    datetime = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    datetime = datetime[2:]
    return datetime


#%%
def kernel_difference_center_meanring(radius: int) -> np.ndarray:
    """Generate kernel that subtracts the center pixel value of the mean of all pixels at a specified distance = `radius`, i.e. a ring around the center pixel with radius = `radius`.

    Parameters
    ----------
    radius : int
        Eculidean distance of the ring to the center pixel.

    Returns
    -------
    np.ndarray
        Kernel of size (`radius*2+1`,`radius*2+1`) with center pixel = -1 @ kernel[radius, radius] and ring for mean creation at distance = `radius ` from center pixel.
    """
    mid = radius
    diameter = radius * 2 + 1

    distances = np.indices((diameter, diameter)) - np.array([mid, mid])[:, None, None]
    distances = np.linalg.norm(distances, axis=0)

    kernel = (np.abs(distances - mid) < 0.5).astype(int)

    kernel = kernel / np.sum(kernel.flatten())
    kernel[mid, mid] = -1


    return kernel

#%%
def channels_extend(
    channels: Dict,
    extend_by: List = ['color_ratios', 'veg_idxs', 'hsl', 'avg', 'decays'],
    ) -> None:


    with np.errstate(divide='ignore', invalid='ignore'):  # Avoid division by zero warnings, in this case NaNs will be assigned
        
        # Color ratios normalized to NIR
        if 'color_ratios' in extend_by:
            
            # Define mask where division by zero and resulting NaN will be replaced with assignment of zero
            mask_zero = channels['nir'] == 0
            
            channels['nblue'] = channels['blue'] / channels['nir']
            channels['nblue'][mask_zero] = 0
            
            channels['ngreen'] = channels['green'] / channels['nir']
            channels['ngreen'][mask_zero] = 0
           
            channels['nred'] = channels['red'] / channels['nir']
            channels['nred'][mask_zero] = 0
            
            channels['nre'] = channels['re'] / channels['nir']
            channels['nre'][mask_zero] = 0
            
        # Vegetation indices
        if 'veg_idxs' in extend_by:
            
            mask_zero = (channels['nir'] + channels['red']) == 0
            channels['ndvi'] = (channels['nir'] - channels['red']) / (channels['nir'] + channels['red'])
            channels['ndvi'][mask_zero] = 0
            
            mask_zero = (channels['re'] + channels['red']) == 0
            channels['ndvire'] = (channels['re'] - channels['red']) / (channels['re'] + channels['red'])
            channels['ndvire'][mask_zero] = 0
            
            mask_zero = (channels['nir'] + channels['re']) == 0
            channels['ndre'] = (channels['nir'] - channels['re']) / (channels['nir'] + channels['re'])
            channels['ndre'][mask_zero] = 0
            
            mask_zero = (channels['green'] + channels['red']) == 0
            channels['grvi'] = (channels['green'] - channels['red']) / (channels['green'] + channels['red'])
            channels['grvi'][mask_zero] = 0
            
            mask_zero = (channels['nir'] + 6 * channels['red'] - 7.5 * channels['blue'] + 1) == 0
            channels['evi'] = 2.5 * (channels['nir'] - channels['red']) / (channels['nir'] + 6 * channels['red'] - 7.5 * channels['blue'] + 1)
            channels['evi'][mask_zero] = 0
            
    # HLS color space
    if 'hsl' in extend_by:
    
        rgb = np.empty((channels['blue'].shape[0], channels['blue'].shape[1], 3), dtype=np.float32)
        rgb[:,:,0] = channels['red']
        rgb[:,:,1] = channels['green']
        rgb[:,:,2] = channels['blue']

        hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)
        channels['hue'] = hls[:,:,0]
        channels['light'] = hls[:,:,1]
        channels['sat'] = hls[:,:,2]
    
    # Average over all color channels
    if 'avg' in extend_by:

        all = np.empty((channels['blue'].shape[0], channels['blue'].shape[1], 5), dtype=np.float32)
        all[:,:,0] = channels['blue'].copy()
        all[:,:,1] = channels['green'].copy()
        all[:,:,2] = channels['red'].copy()
        all[:,:,3] = channels['re'].copy()
        all[:,:,4] = channels['nir'].copy()
  
        channels['avg'] = np.mean(all, axis=2)
    
    
    # Height decay around center pixel
    if 'decays' in extend_by:
        
        for radius in [int(2**i) for i in range(0,5)]:
            
            img = channels['chm'].copy()
            img[~np.isfinite(img)] = 0

            kernel = kernel_difference_center_meanring(radius)
            channels[f"decay{radius}"] = cv2.filter2D(
                src = img,
                ddepth = -1,
                kernel = kernel,
                borderType = cv2.BORDER_REFLECT,
                )
                
    pass

#%%
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
    vmin = np.nanpercentile(rgb, perc)
    vmax = np.nanpercentile(rgb, 100 - perc)

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


def affine_numpy_to_resterio(affine_numpy: np.ndarray) -> rasterio.Affine:
    """ Transform affine geo transformation from np.ndarray of shape (3,3) to rasterio.Affine object.

    Parameters
    ----------
    affine_numpy : np.ndarray
        Affine geo transformation of shape (3,3) of order: `[[a, b, c], [d, e, f], [g, h, i]]`, see rasterio.Affine.

    Returns
    -------
    rasterio.Affine
        Affine geo transformation
    """
    affine = affine_numpy.copy()
    affine = affine.flatten()
    affine_rasterio = rasterio.Affine(affine[0], affine[1], affine[2], affine[3], affine[4], affine[5])

    return affine_rasterio


def geojson_records_fields_to_numpy_dtype(fields, exclude_names: List[str] = None) -> np.dtype:

    # Exclude some fields according to exclude_names
    include_names = [f[0] for f in fields[1:] if f[0] not in exclude_names]

    # Initialize dtypes with 'id' field
    dtypes = [('id', 'u2')]

    for f in fields[1:]: 
        name = f[0]

        if name in include_names:

            # Convert name to lower case
            name = name.lower()
            
            # Define dtype cases
            is_int = (f[1] == 'N') & (f[3] == 0)  # Integer
            is_float = (f[1] == 'F') & (f[3] > 0) # Float
            is_str = (f[1] == 'C') & (f[3] == 0)  # String

            # Assign dtype according to cases
            if is_int:  # Integer dtype
                dtype = 'i4'
            elif is_float:
                dtype = 'f4'
            elif is_str:
                dtype = f"S{f[2]}"
            else:
                print(f"Field {f[0]} was not included, no numpy.dtype equvalent found for ({f[1]},{f[2]},{f[3]})")
                dtype = None
            
            if dtype != None:
                dtypes.append((name, dtype))

    return np.dtype(dtypes)


#%%
def labelimage_to_boundsimage(labels: np.ndarray) -> np.ndarray:
    """Return boundaries of labels.

    Parameters
    ----------
    labels : np.ndarray
        Image where connected components are assigned with one unique integer, i.e. labels.

    Returns
    -------
    np.ndarray
        Boundaries of all labels as np.uint8 array. Boundaries are assigned with 1, rest is zero.
    """
    bounds = skimage.segmentation.find_boundaries(
        labels,
        connectivity=1,
        mode='outer',
    ).astype(np.uint8)
    bounds = skimage.morphology.thin(bounds)

    return bounds


#%%
def labelimage_to_polygons(img: np.ndarray) -> Dict:

    # Define nodata mask
    mask = img > 0

    # Get shapes in image using rasterio.features.shapes()
    shapes = rasterio.features.shapes(img, mask = mask)

    # Transform shapes to standardized polygons format
    polygons_dict = {}
    for shape in shapes:
        idx = int(shape[1])
        poly = np.array(shape[0]['coordinates'][0], dtype=np.float32)
        polygons_dict[idx] = poly

    # Sort polygons dict by key
    polygons_dict = dict([(key, polygons_dict[key]) for key in sorted(polygons_dict.keys())])

    return polygons_dict


#%%
def polygons_to_labelimage(polygons: Dict, shape: Tuple):

    # Init labelimage
    labelimg = np.zeros(shape, dtype=np.uint16)

    # Prepare shapes as List of (poly, id) for rasterio.features.rasterize()
    shapes = []
    for id, poly in polygons.items():
    
        # Convert poly to shapely poly
        poly = shapely.geometry.Polygon(poly)

        # Convert shapely poly to geojson poly
        poly = shapely.geometry.mapping(poly)

        # Extend shapes with (poly, id) Tuple
        shapes.extend([(poly, id)])

    # Now burn shapes into empty labelimg
    rasterio.features.rasterize(shapes, out = labelimg)

    return labelimg


#%%
def labelimage_to_idxlist(labels_in: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    '''
    Return indices of all labels as list.

    Parameters:
    -----------
    labels_in: np.ndarray
        Integer labeled connected components of an image (see skimage.measure.labels).
    
    Returns:
    --------
    List of ``len(labels)``. Each entry corresponds to ``Tuple[np.ndarray,np.ndarray]`` containing (coordinate) indices of all pixels belonging to unique label in original image.

    '''
    shape = labels_in.shape # Original shape of labels, corresponds to shape of image
    labels = labels_in.flatten() # Flatten

    # Get number of unique counts in labels plus the sort-index of the labels to reconstruct indices of each label.
    labels_len = np.unique(labels, return_counts=True)[1]
    labels_sortidx = np.argsort(labels)

    # Now loop through each label and get indices
    labels_idx = []
    i0 = 0
    for l in labels_len:
        i1 = i0 + l
        label_idx = labels_sortidx[i0:i1]
        i0 = i1

        # Go from flattened index to coordinate index
        label_idx = np.unravel_index(label_idx, shape) 
        
        labels_idx.append(label_idx)

    labels_idx = labels_idx[1:] # Remove index belonging to background (i.e. labels==0)

    return labels_idx