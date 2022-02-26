import numpy as np
import cv2 as cv

def blur(img, kernel_size=3, iterations=3):
    img_blur = img.copy()

    for i in range(iterations):
        img_blur = cv.medianBlur(img_blur,kernel_size)
    for i in range(iterations):
        img_blur = cv.GaussianBlur(img_blur,(kernel_size,kernel_size),0)
    return img_blur


def divide_connected_regions(mask, percentile=25):
    
    # Get connected regions in mask -> isles
    count, isles = cv.connectedComponents(mask.astype(np.uint8))
    isles = isles.flatten() # Flatten

    # Get flat-indices of isles in original mask
    isles_len = np.unique(isles, return_counts=True)[1]
    isles_sidx = np.argsort(isles)

    isles_midx = []
    i0 = 0
    for l in isles_len:
        i1 = i0 + l
        isle_midx = isles_sidx[i0:i1]
        i0 = i1

        isles_midx.append(isle_midx)

    isles_midx = isles_midx[1:] # Remove index belonging to background (value of zero in cv.connectedComponents)

    # Distance trafo of mask
    dist_trafo = cv.distanceTransform(mask.astype(np.uint8), cv.DIST_L2, 5)

    # Initiate new mask
    new_mask = dist_trafo.astype(np.float32)
    new_mask = new_mask.flatten() # Flatten to use isles_midx
    
    # Go through isles and create sub-isles
    for idx in isles_midx:
        crit = np.percentile(new_mask[idx], percentile)
        isle_max = np.max(new_mask[idx])
        
        if isle_max > crit:
            new_mask[idx] = new_mask[idx] - crit

    new_mask[new_mask <= 0] = 0
    new_mask[new_mask > 0] = 1

    return new_mask.astype(np.uint8).reshape(mask.shape)