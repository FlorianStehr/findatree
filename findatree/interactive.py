from typing import Dict, List, Tuple
import numpy as np
import cv2
import skimage.exposure

# bokeh modules
import bokeh.plotting
import bokeh.models

# bokeh classes
from bokeh.plotting.figure import Figure


#%%
def _img_RGBfloat32_to_RGBAuint32(rgb:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert RGB float32 image of shape (M,N,3) to RGBA uin32 image of shape (M,N) for use in `image_rgba()` of bokeh.

    Parameters
    ----------
    rgb : np.ndarray
        RGB float32 image of shape (M,N,3)

    Returns
    -------
    Tuple[np.ndarray,np.ndarray]
        rgba_uint32: np.ndarray
            RGBA uint32 image of shape (M,N)
        rgba_uint8: np.ndarray
            RGBA uint8 image of shape (M,N,4)
    """
    
    # Assertions
    assert rgb.ndim == 3, 'rgb.ndim must be 3'
    assert rgb.dtype == np.float32, 'rgb.dtype must be np.float32'
    assert rgb.shape[-1] == 3, 'rgb.shape[-1] must be 3'
    
    shape = rgb.shape

    # Get image percentiles
    p = 1
    vmin = np.nanpercentile(rgb, p)
    vmax = np.nanpercentile(rgb, 100-p)

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
    
    return rgba_ui32, rgba_ui8

#%%

def bk_fig_img_rgb(img_in:np.ndarray) -> Tuple:

    # Convert RGB image to correct format
    rgba_ui32, rgba_ui8 = _img_RGBfloat32_to_RGBAuint32(img_in)
    shape = rgba_ui32.shape

    # Define source
    data = {
        'rgba': [rgba_ui32],
        'red': [rgba_ui8[:,:,0]],
        'green': [rgba_ui8[:,:,1]],
        'blue': [rgba_ui8[:,:,2]],
        'x': [0],
        'y': [shape[0]],
        'dw': [shape[1]],
        'dh': [shape[0]],
    }
    source = bokeh.plotting.ColumnDataSource(
        data=data,
    )

    # Define figure
    fig = bokeh.plotting.figure(
        x_range=(0, shape[1]),
        y_range=(shape[0], 0),
        active_scroll='wheel_zoom',
    )

    # Add image
    img_rgba = fig.image_rgba(
        source=source,
        image='rgba',
        x='x',
        y='y',
        dw='dw',
        dh='dh')

    # Add tools
    hover_tool = bokeh.models.HoverTool(
        tooltips=[
            ("(x, y)", "($x, $y)"),
            ("RGB", "(@red, @green, @blue)")]
        )
    fig.add_tools(hover_tool)

    # Activated/Deactivated tools
    fig.toolbar.active_inspect = None

    return fig

####################################################################################################
####################################################################################################
####################################################################################################

# import bokeh.plotting
# import bokeh.models
# import bokeh.transform
# import bokeh.layouts

# importlib.reload(interact)

# fig_rgba = interact.bk_fig_img_rgb(cs['RGB'])

# shape = params_cs['shape']
# data = {
#         'chm': [cs['chm'].copy()],
#         'ndvi': [cs['ndvi'].copy()],
#         'lightness': [cs['l'].copy()],
#         'x': [0],
#         'y': [shape[0]],
#         'dw': [shape[1]],
#         'dh': [shape[0]],
# }
# source = bokeh.plotting.ColumnDataSource(data=data)

# panels = []
# for key in ['chm', 'ndvi', 'lightness']:
#     # Figure
#     fig = bokeh.plotting.figure(
#             x_range=fig_rgba.x_range,
#             y_range=fig_rgba.y_range,
#             active_scroll='wheel_zoom',
#     )

#     # Add colormapper
#     p = 0
#     vmin = np.nanpercentile(source.data[key],p)
#     vmax = np.nanpercentile(source.data[key],100 - p)
#     mapper = bokeh.transform.LinearColorMapper(palette='Greys256' , low=vmin , high=vmax)

#     # Add image
#     img = fig.image(
#         source=source,
#         image=key,
#         x='x',
#         y='y',
#         dw='dw',
#         dh='dh',
#         color_mapper=mapper,
#     )

#     # Add slider
#     slider = bokeh.models.RangeSlider(start=vmin, end=vmax, value=(vmin,vmax), step=(vmax-vmin)/20, title=f"{key} range:")
#     slider.js_link('value',mapper,'low', attr_selector=0)
#     slider.js_link('value',mapper,'high', attr_selector=1)

#     # Layout
#     col = bokeh.layouts.column([fig, slider])
#     panel = bokeh.models.Panel(child=col, title=key)
#     panels.extend([panel])

# tabs = bokeh.models.Tabs(tabs=panels, tabs_location = 'right')
# rows = bokeh.layouts.row([fig_rgba, tabs])

# show(rows)

####################################################################################################
####################################################################################################
####################################################################################################

# from bokeh.layouts import layout
# from bokeh.transform import linear_cmap
# from bokeh.models import RangeSlider, Slider


# p1 = figure()
# img_rgba = p1.image_rgba(
#     image=[rgba_ui32],
#     x=[0],
#     y=[0],
#     dw=[shape[0]],
#     dh=[shape[1]],
# )

# p2 = figure(x_range=p1.x_range, y_range=p1.y_range)

# rslider_chm = RangeSlider(start=0, end=50, value=(5,30), step=1, title='CHM height')
# color_mapper = LinearColorMapper(palette="Greys256", low=0, high=40)
# rslider_chm.js_link('value',color_mapper,'low',attr_selector=0)
# rslider_chm.js_link('value',color_mapper,'high',attr_selector=1)

# p2.image(
#     image=[cs['chm']],
#     x=[0],
#     y=[0],
#     dw=[shape[0]],
#     dh=[shape[1]],
#     color_mapper=color_mapper,
# )

# lo = layout([
#     [rslider_chm],
#     [
#         [p1],
#         [p2]
#     ]
# ])

# show(lo)