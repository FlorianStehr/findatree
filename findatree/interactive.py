from turtle import width
from typing import Dict, List, Tuple
from matplotlib.pyplot import colorbar, figimage
import numpy as np
import importlib

import bokeh.plotting
import bokeh.models
import bokeh.layouts

from bokeh.plotting.figure import Figure
from bokeh.models import Column
from bokeh.models import Toggle

import findatree.transformations as transformations
importlib.reload(transformations)

#%%
class Plotter:
    
    def __init__(self):
        
        self.channels = None
        self.segments = None
        self.source = bokeh.plotting.ColumnDataSource(data={})
        self.figures = {}
        
        # Style attributes
        self.size = 400
        self.palette = 'Magma256'
        self.title_font_size = '150%'
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

        try:
            data[channel_name] = [self.channels[channel_name].copy()]
        except:
            try:
                data[channel_name] = [self.segments[channel_name].copy()]
            except:
                raise KeyError(f"`{channel_name}` not in self.channels nor in self.segments.")

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
            width = self.size,
            height = self.size,
            x_range = self.x_range,
            y_range = self.y_range,
            x_axis_label='x [m]',
            y_axis_label='y [m]',
            active_scroll='wheel_zoom',
            )

        # Pass on x_range to other figures for same FOV
        self.x_range = fig.x_range
        self.y_range = fig.y_range
        
        return fig

#%%
    def figures_add_rgb(self, perc:float = 1):
        
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

        # Update
        self.figures['rgb'] = [fig]

        pass
        

#%%
    def figures_add_gray(self, channel_name: str, perc: float = 0.5):
        
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
        
        # Update
        self.figures[channel_name] = [fig, slider]
        
        pass

#%%
    def _figures_add_toggler_bounds(self) -> Toggle:
        
        # Assert that bounds are in source
        assert 'bounds' in self.source.data.keys(), "`bounds` not in self.source.data"
            
        # Add toggler for showing bounds
        toggler = bokeh.models.Toggle(
            active = True, 
            label = "Show Segments",
            button_type = "success",
            width = self.size // 4,
            height = self.size // 8,
            )
        
        # Loop through all figures and add bounds image connected to toggler
        for key in self.figures:
            fig = self.figures[key][0]

            # Add bounds
            bounds = fig.image(
                source=self.source,
                image='bounds',
                x='x',
                y='y',
                dw='width',
                dh='height',
                color_mapper=self.bounds_color_mapper,
                visible=True,
            )
            toggler.js_link('active', bounds, 'visible')

        return toggler


#%%
    def create_layout(self):
        
        # Create list of bokeh columns by combining all elements in self.figures and add title division
        cols = []
        for key in self.figures:
            # Add title division
            div = bokeh.models.Div(
                text=f"{key}".upper(),
                style={'font-size': self.title_font_size},
                width=self.size,
                height=self.size // 16,
                )

            # Define column elements as list 
            col_elements = [div]
            for item in self.figures[key]:
                col_elements.extend([item])

            # Create column and add to columns  
            col = bokeh.layouts.column(col_elements)
            cols.extend([col])

        # Create final layout
        try:
            toggler = self._figures_add_toggler_bounds()
            cols.extend([toggler])
        except:
            pass

        layout = bokeh.layouts.layout([cols])
        
        return layout 