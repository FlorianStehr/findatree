from typing import List
from typing import Tuple
import numpy as np


#%%
def labels_idx(labels_in: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
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

    # Now loop through each label and get indice
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
