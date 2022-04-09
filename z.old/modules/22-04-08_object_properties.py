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