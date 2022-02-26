import numpy as np
import matplotlib.pyplot as plt

from typing import List


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
    boolmask: np.ndarray(shape=(m,n), dtype=np.bool_):
        Channel indices where ``boolmask == True`` will be set to 0. Must be of shape ``(m,n)``. De
    contrasts: List[Tuple, ...]
        List of tuples containing contrast lower and upper threshold for each channel (i.e. ``vmin``, ``vmax`` of matplotlib.pyplot.imshow()).
        Defaults to ``[(np.nanpercentile(c.flatten(), 1), np.nanpercentile(c.flatten(), 99)) for c in channels]`` after mask was applied.
    channel_names: List[str]
        List of channel names to be shwon as title of each axis in figure.
    zoom: float
        By default single ax will be of size ``(3,3)`` [inches]. Zoom scales this factor to ``(3 * zoom, 3 * zoom)``.




    
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
    if 'boolmask' not in kwargs:
        boolmask = np.zeros(shape, dtype=np.bool_)
    else:
        boolmask = kwargs['boolmask'].astype(np.bool_)
    #
    if 'xylim' not in kwargs:
        xylim = [(s // 2, s // 2) for s in shape[::-1]] # (center, width)
    else:
        xylim = kwargs['xylim']
    #
    if 'channel_names' not in kwargs:
        channel_names = [None for i in range(len(channels))]
    else:
        channel_names = kwargs['channel_names']
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
    boolmask = boolmask[rmin:rmax, cmin:cmax]

    # Apply mask
    channels_mask = []
    for c in channels:
        c_mask = c.copy().astype(np.float32)
        c_mask[boolmask] = np.nan
        channels_mask.append(c_mask)

    # Now get default contrast values
    if 'contrasts' not in kwargs:
        contrasts = [(np.nanpercentile(c.flatten(), 1), np.nanpercentile(c.flatten(), 99)) for c in channels]
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
    f.subplots_adjust(bottom=0.05, top=0.95, left=0, right=1, hspace=0, wspace=0)

    for i, ax in enumerate(axs.flatten()):
        if i < len(channels):

            if contrasts[i] is None:
                vmin = np.nanpercentile(channels_mask[i].flatten(), 1)
                vmax = np.nanpercentile(channels_mask[i].flatten(), 99)
            else:
                vmin = contrasts[i][0]
                vmax = contrasts[i][1]

            mapp = ax.imshow(
                channels_mask[i],
                vmin=vmin,
                vmax=vmax,
                cmap='magma',
            )
            
            ax.set_xticks(np.arange(0, cmax - cmin, 200))
            ax.set_xticklabels(np.arange(cmin / 100, cmax / 100, 2, dtype=np.uint8))
            ax.set_yticks(np.arange(0, rmax - rmin, 200))
            ax.set_yticklabels(np.arange(rmin / 100, rmax / 100, 2, dtype=np.uint8))

            if (len(channel_names) == len(channels)):
                ax.set_title(f"{channel_names[i]} [{vmin:.0e} - {vmax:.0e}]")
            else:
                ax.set_title(f"[{vmin:.0e} - {vmax:.0e}]")

        else:
            ax.axis('off')

    return f, axs.flatten()