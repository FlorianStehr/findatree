import numpy as np
import matplotlib.pyplot as plt

from typing import List


def _image_minmax(img, lower_perc: int=0, upper_perc: int=100):

    # Get flattened version of image without nans
    x = img[np.isfinite(img)]
    # Get unique values
    x_uni = np.unique(x)
    
    if len(x_uni) == 0: # Only one unique value case
        xmin = x_uni
        xmax = x_uni
        return xmin, xmax

    elif 0 < len(x_uni) <= 10: # If less than 10 unique values return minimum, maximum
        xmin = x_uni[0]
        xmax = x_uni[-1]
        return xmin, xmax

    else: # If more than 10 unique values return percentiles
        xmin = np.percentile(x, lower_perc)
        xmax = np.percentile(x, upper_perc)
        return xmin, xmax
    

def show_channels(
    channels: List[np.ndarray],
    **kwargs,
):  
    '''
    Plotting function for channels

    Parameters:
    -----------
    channels: List[np.ndarray]
        All color channels to be plotted as np.ndarray of shape ``(m,n)``. All channels are assumed to have eaul dimensions.

    Other Parameters:
    -----------------
    n_cols_rows: List
        Number of colums and rows in figure. Defaults to ``[len(channels), 1]``.
    xylim: List[Tuple, Tuple]
        x-y-limits given as ``[(x_center, x_width), (y_center, y_width)]``. Defaults to cover orginal dimensions of channels.
    mask: np.ndarray(shape=(m,n), dtype=np.bool_):
        Channel indices where ``mask == False`` will be set to 0. Must be of shape ``(m,n)``. De
    contrasts: List[Tuple, ...]
        List of tuples containing contrast lower and upper threshold for each channel (i.e. ``vmin``, ``vmax`` of matplotlib.pyplot.imshow()).
        Defaults to ``[(np.nanpercentile(c.flatten(), 1), np.nanpercentile(c.flatten(), 99)) for c in channels]`` after mask was applied.
    channel_names: List[str, ...]
        List of channel names to be shwon as title of each axis in figure.
    zoom: float
        By default single ax will be of size ``(3,3)`` [inches]. Zoom scales this factor to ``(3 * zoom, 3 * zoom)``.
    use_random_cmap: List[bool, ...]
        Use random colormap for channel. Defaults to ``False`` for each channel and then matplotlib cmap ``magma`` is used. If random colormap is used input image will be transformed by ``img % 256``.




    
    Returns:
    --------
    f:
        Figure object as returned by matplotlib.pyplot.subplots()
    axs:
        Flattened axes as returned by matplotlib.pyplot.subplots()

    '''
    shape = channels[0].shape

    # Check for kwargs and assign
    if 'n_cols_rows' not in kwargs:
        n_cols_rows = [len(channels), 1]
    else:
        n_cols_rows = kwargs['n_cols_rows']
    #
    if 'mask' not in kwargs:
        mask = np.ones(shape, dtype=np.bool_)
    else:
        mask = kwargs['mask'].astype(np.bool_)
    #
    if 'xylim' not in kwargs:
        xylim = [(s // 2 + 1, s // 2 + 1) for s in shape[::-1]] # (center, width)
    else:
        xylim = kwargs['xylim']
    #
    if 'channel_names' not in kwargs:
        channel_names = [None for i in range(len(channels))]
    else:
        channel_names = kwargs['channel_names']
    #
    if 'use_random_cmap' not in kwargs:
        use_random_cmap = [False for i in range(len(channels))]
    else:
        use_random_cmap = kwargs['use_random_cmap']
    #
    if 'zoom' not in kwargs:
        zoom = 1
    else:
        zoom = kwargs['zoom']
    
    # Reduce channels & mask to xylim
    rmin = xylim[1][0] - xylim[1][1]
    rmax = xylim[1][0] + xylim[1][1]
    cmin = xylim[0][0] - xylim[0][1]
    cmax = xylim[0][0] + xylim[0][1]
    channels = [c[rmin:rmax, cmin:cmax] for c in channels]
    mask = mask[rmin:rmax, cmin:cmax]

    # Apply mask
    channels_mask = []
    for c in channels:
        c_mask = c.copy().astype(np.float32)
        c_mask[np.invert(mask)] = np.nan
        channels_mask.append(c_mask)

    # Now get default contrast values
    if 'contrasts' not in kwargs:
        contrasts = [_image_minmax(c_mask) for c_mask in channels_mask]
    else:
        contrasts = kwargs['contrasts']

    # Plotting
    f, axs = plt.subplots(
        ncols=n_cols_rows[0],
        nrows=n_cols_rows[1],
        sharex='all',
        sharey='all',
        figsize=(n_cols_rows[0] * 3 * zoom, n_cols_rows[1] * 3 * zoom),
        tight_layout=True,
        )
    f.subplots_adjust(bottom=0.05, top=0.95, left=0, right=1, hspace=0.15, wspace=0)

    # Define random cmap
    vals = np.linspace(0.05,1,256)
    np.random.shuffle(vals)
    vals[0] = 0
    cmap_random = plt.cm.colors.ListedColormap(plt.cm.nipy_spectral(vals))
    cmap = 'magma'

    for i, ax in enumerate(axs.flatten()):
        if i < len(channels):
            img = channels_mask[i].astype(np.float32)

            if contrasts[i] is None:
                vmin, vmax = _image_minmax(img)
            else:
                vmin = contrasts[i][0]
                vmax = contrasts[i][1]

            if use_random_cmap[i]:
                cmap = cmap_random
                img[img > 0] = img[img > 0] % 256 + img[img > 0] // 256
                img[img > 256] = 100
                vmin, vmax = _image_minmax(img,0,100)

            mapp = ax.imshow(
                img,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )

            tick = 10 ** np.floor(np.log10(cmax - cmin))
            if ((cmax - cmin) / tick >= 1) & ((cmax - cmin) / tick <= 2):
                tick = (tick / 10) * 2.5
            ax.set_xticks(np.arange(0, cmax - cmin, tick))
            
            tick = 10 ** np.floor(np.log10(rmax - rmin))
            if ((rmax - rmin) / tick >= 1) & ((rmax - rmin) / tick <= 2):
                tick = (tick / 10) * 2.5
            ax.set_yticks(np.arange(0, rmax - rmin, tick))

            if (len(channel_names) == len(channels)):
                ax.set_title(f"{channel_names[i]} [{vmin:.0e} - {vmax:.0e}]")
            else:
                ax.set_title(f"[{vmin:.0e} - {vmax:.0e}]")

        else:
            ax.axis('off')

    return f, axs.flatten()