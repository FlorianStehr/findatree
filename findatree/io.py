from typing import Dict, List, Tuple
import os
import h5py

#%%
def channels_to_hdf5(channels: Dict , params_channels: Dict, dir_name: str=r'C:\Data\lwf\processed') -> None:
    """Save all secondary channels in .hdf5 container.

    Group `channels` wil be created in .hdf5 with `channels` as datasets and `params_channels` as group attributes.
    If .hdf5 already exists and dsm, dtm and ortho paths did not change the file is overwritten, otherwise an error is raised.

    Parameters
    ----------
    channels : Dict
        Dict. of all normalized secondary channels as returned by findatree.geo_to_image.channels_primary_to_secondary()).
    params_channels : Dict
        Dict. of parameters as returned by geo_to_image.findatree.load_channels().
    dir_name: str
        Path to directory where .hdf5 is stored.
    """
    # Define name of .hdf5
    name = f"tnr{params_channels['tnr']}.hdf5"
    
    # Define full path to .hdf5, i.e. directory + name
    path = os.path.join(dir_name, name)

    # Open file for writing if exists, create otherwise.
    with h5py.File(path, 'a') as f:
        '''Three possible cases to cover:
            1. Channels group EXISTS ...
                a. ... and dsm, dtm and ortho path attributes are NOT the same as in params_channels -> Raise Error
                b. ... and dsm, dtm and ortho path attributes ARE the same as in params_channels -> Continue and overwrite 
            2. Channels group does NOT EXIST -> Write
        '''
        if 'channels' in f:
            grp = f.get('channels')
            
            # Assert that any of the dsm, dtm and ortho paths saved in group attributes are the same as in params_channels
            for key in ['path_dsm', 'path_dtm', 'path_ortho']:
                assert grp.attrs[key] == params_channels[key], f"You are trying to overwrite channels with a different `{key}` attribute. No file was written. Change the saving directory."
        
        else:
            grp = f.create_group('channels')
        
        # Update all parameters as group attributes
        for key in params_channels:
            grp.attrs[key] = params_channels[key]

        # Add/update all channel as group datasets
        for key in channels:
            
            # Overwrite case [1b]
            if key in grp: 
                del grp[key]
                grp.create_dataset(key, data=channels[key])
            
            # Write case [2]
            else:
                dset = grp.create_dataset(key, data=channels[key])
        pass


#%%
def segments_to_hdf5(segments: Dict , params_segments: Dict, dir_name: str=r'C:\Data\lwf\processed') -> None:
    """Save all segmentation maps in .hdf5 container.

    Group `segments` wil be created in .hdf5 with `channels` as datasets and `params_segments` as group attributes.

    Parameters
    ----------
    segments : Dict
        Dictionary of labels as returned by segmentation.segment()
    params_segments : Dict
        Dictionary ofsegmentation parameters as returned by segmentation.segment()
    dir_name: str
        Path to directory where .hdf5 is stored
    """
    # Define name of .hdf5
    name = f"tnr{params_segments['tnr']}.hdf5"
    
    # Define full path to .hdf5, i.e. directory + name
    path = os.path.join(dir_name, name)

    # Open file for writing if exists, create otherwise.
    with h5py.File(path, 'a') as f:
        '''Two possible cases to cover:
            1. Segments group EXISTS -> Continue and overwrite 
            2. Segments group does NOT EXIST -> Write
        '''
        if 'segments' in f:
            grp = f.get('segments')
        else:
            grp = f.create_group('segments')
        
        # Update all parameters as group attributes
        for key in params_segments:
            grp.attrs[key] = params_segments[key]

        # Add/update all channel as group datasets
        for key in segments:

            # Overwrite case [1]
            if key in grp: 
                del grp[key]
                grp.create_dataset(key, data=segments[key])
            
            # Write case [2]
            else:
                dset = grp.create_dataset(key, data=segments[key])
        pass