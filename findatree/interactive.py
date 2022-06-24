from turtle import width
from typing import Dict, List, Tuple
from matplotlib.pyplot import colorbar
import numpy as np
import importlib

import bokeh.plotting
import bokeh.models
import bokeh.layouts

from bokeh.plotting.figure import Figure
from bokeh.models import Column

import findatree.transformations as transformations
importlib.reload(transformations)

#%%
class Plotter:
    
    def __init__(self):
        
        self.channels = None
        self.segments = None
        self.source = bokeh.plotting.ColumnDataSource(data={})
        self.figures = []
        
        # Style attributes
        self.size = 400
        self.palette = 'Magma256'
        self.nan_color = 'rgba(0, 0, 0, 0)'
        self.bounds_color_mapper = bokeh.transform.LinearColorMapper(
            palette='Greys8',
            low=1,
            high=1,
            nan_color=self.nan_color,
        )

        pass

#%%
    def add_channels(self, channels: Dict, params_channels: Dict) -> None:

        # Get source data
        data = self.source.data
        
        # Add dimensions in meters to source
        data['x'] = [0]
        data['y'] = [params_channels['shape'][0] * params_channels['px_width']]
        data['width'] = [params_channels['shape'][1] * params_channels['px_width']]
        data['height'] = [params_channels['shape'][0] * params_channels['px_width']]

        # Add empty (i.e. NaN values) segment bounds to source
        data['bounds'] = [np.ones(params_channels['shape'], dtype=np.float32) * np.nan]

        # Update
        self.source = bokeh.plotting.ColumnDataSource(data=data)
        self.channels = channels

        pass


#%%
    def add_segments(self, segments: Dict) -> None:

        # Get source data
        data = self.source.data

        # Convert boundaries to float32 with NaNs @ negatives
        bounds = segments['bounds'].copy()
        bounds = bounds.astype(np.float32)
        bounds[bounds == 0] = np.nan

        # Add boundaries to source
        data['bounds'] = [bounds]

        # Update
        self.source = bokeh.plotting.ColumnDataSource(data=data)
        self.segments = segments

        pass

#%%
    def _source_add_channel(self, channel_name):
        
        # Get source data
        data = self.source.data

        data[channel_name] = [self.channels[channel_name].copy()]

        # Update
        self.source = bokeh.plotting.ColumnDataSource(data=data)

        pass    

#%%
    def _source_add_rgb(self, perc):
        
        data = self.source.data

        # Add RGB channel in uint32 RGBA format
        red = self.channels['red']
        green = self.channels['green']
        blue = self.channels['blue']
        rgb, rgba = transformations.rgb_to_RGBA(red, green, blue, perc)
        data['rgb'] = [rgba]

        # Update
        self.source = bokeh.plotting.ColumnDataSource(data=data)

        pass

#%%
    def _figure_create(self) -> Figure:

        # Set xy_range for figure
        if len(self.figures) == 0:  # NO other figure -> set xy_range according to source
            self.x_range = (0, self.source.data['width'][0])
            self.y_range = (self.source.data['height'][0], 0)

        # Create figure
        fig = bokeh.plotting.figure(
            width=self.size,
            height=self.size,
            x_range=self.x_range,
            y_range=self.y_range,
            active_scroll='wheel_zoom',
            )

        # Pass on x_range to other figures for same FOV
        self.x_range = fig.x_range
        self.y_range = fig.y_range
        
        return fig

#%%
    def figures_add_rgb(self, perc:float = 1) -> Column:
        
        # Add figure
        fig = self._figure_create()

        # Add rgb channel to source
        self._source_add_rgb(perc)

        # Add rgb image to figure
        img_rgba = fig.image_rgba(
            source=self.source,
            image='rgb',
            x='x',
            y='y',
            dw='width',
            dh='height',
        )

        # Add hover tool to rgb image
        hover_tool = bokeh.models.HoverTool(
            tooltips=[("(x, y)", "($x, $y)")],
        )
        fig.add_tools(hover_tool)

        # Add bounds
        bounds = fig.image(
            source=self.source,
            image='bounds',
            x='x',
            y='y',
            dw='width',
            dh='height',
            color_mapper=self.bounds_color_mapper,
            visible=False,
        )

        # Add toggler for showing bounds
        toggler = bokeh.models.Toggle(
            label="Segments",
            button_type="success",
            width=self.size // 2,
            )
        toggler.js_link('active', bounds, 'visible')

        # Combine figure and toggler to column 
        col = bokeh.layouts.column([fig, toggler])

        # Update
        self.figures.extend([col])
        
        return col

#%%
    def figures_add_gray(self, channel_name: str, perc: float = 1) -> Column:
        
        # Add figure
        fig = self._figure_create()
        
        # Add channel to source
        self._source_add_channel(channel_name)

        # Add colormapper
        vmin = np.nanpercentile(self.source.data[channel_name][0], perc)
        vmax = np.nanpercentile(self.source.data[channel_name][0], 100 - perc)
        color_mapper = bokeh.transform.LinearColorMapper(palette=self.palette , low=vmin , high=vmax, nan_color=self.nan_color)

        # Add gray scale image to figure
        img_rgba = fig.image(
            source=self.source,
            image=channel_name,
            x='x',
            y='y',
            dw='width',
            dh='height',
            color_mapper=color_mapper,
        )

        # Add hover tool
        hover_tool = bokeh.models.HoverTool(
            tooltips=[
                ("(x, y)", "($x, $y)"),
                (channel_name,f"@{channel_name}"),
            ],
        )
        fig.add_tools(hover_tool)

        # Add  range slider and link to color_mapper range
        slider = bokeh.models.RangeSlider(
            start=vmin,
            end=vmax,
            value=(vmin,vmax),
            step=(vmax-vmin)/50,
            title=f"{channel_name} range:",
            width=self.size // 2,
        )
        slider.js_link('value',color_mapper,'low', attr_selector=0)
        slider.js_link('value',color_mapper,'high', attr_selector=1)

        # Combine figure and slider to column 
        col = bokeh.layouts.column([fig, slider])
        
        # Update
        self.figures.extend([col])
        
        return col

    def create_layout(self):

        # Create final layout
        layout = bokeh.layouts.layout([self.figures])
        
        return layout 



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