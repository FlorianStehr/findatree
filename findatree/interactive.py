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
import findatree.geo_to_image as geo_to_image
importlib.reload(transformations)

#%%
class Plotter:
    
    def __init__(self):
        
        # Sata sources connected to class
        self.channels = None
        self.source = bokeh.plotting.ColumnDataSource(data={})
        self.source_crowns = {}

        # Bokeh artists dictionaries
        self.figures = {}
        self.togglers = {}
        
        # Plot content
        self.channels_downscale = 0
        self.show_features = ['id', 'bnr', 'ba']

        # Style attributes
        self.width = 400
        self.palette = 'Magma256'
        self.title_font_size = '150%'
        self.patches_alpha = 0.5
        
        # Pseudo style attributes that will be automatically updated by function calls
        self.x_range = None # Will be updated during creation of first figure
        self.y_range= None  # Will be updated during creation of first figure
        self.aspect_ratio = 1  # Height/Width, will be updated during creation of first figure
        self.heigth = 400  # Will be updated during creation of first figure

        # Style attributes that are usually not changed
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

        # Downscale channels using gaussian image pyramids
        channels_down, params_channels_down = geo_to_image._channels_downscale(channels, params_channels, downscale = self.channels_downscale)

        # Get source data
        data = self.source.data
        
        # Add dimensions in pxs to source
        data['x'] = [0]
        data['y'] = [ params_channels_down['shape'][0] ]
        data['width'] = [ params_channels_down['shape'][1] ]
        data['height'] = [ params_channels_down['shape'][0] ]

        # Update
        self.source = bokeh.plotting.ColumnDataSource(data=data)
        self.channels = channels_down

        pass


#%%
    def _source_add_channel(self, channel_name):
        
        # Get source data
        data = self.source.data

        try:
            data[channel_name] = [self.channels[channel_name].copy()]
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
    def _source_crowns_add(
        self,
        crowns: Dict,
        params_crowns: str,
        ) -> None:
        
        # Get downscaling factor and y offset due to inverted axis
        downscale = self.channels_downscale
        offset = self.source.data['height'][0]

        # Prepare patches source: Add polygons
        polys = crowns['polygons']
        patches_data = {
            'xs' : [poly[:, 0] / 2**downscale for poly in polys.values()],
            'ys' : [offset - (poly[:, 1] / 2**downscale) for poly in polys.values()],
        }

        # Prepare patches source: Add features
        features_all = crowns['features']
        for name in self.show_features:
            for features in features_all.values():
                try:
                    patches_data[name] = [feature for feature in features[name]]
                except:
                    print(f"Field `{name}` could not be assigned to source")

        # Update source
        self.source_crowns[params_crowns['origin']] = bokeh.plotting.ColumnDataSource(data=patches_data)

        pass

#%%
    def _figure_create(self, name) -> Figure:

        # Set xy_range for figure
        if len(self.figures) == 0:  # NO other figure -> set xy_range according to source
            self.x_range = (0, self.source.data['width'][0])
            self.y_range = (self.source.data['height'][0], 0)
            
            # Update aspect ratio according to ranges the first time
            self.aspect_ratio = self.y_range[0] / self.x_range[1]

            # Update height according to aspect ratio and width
            self.height = int(self.width * self.aspect_ratio)

        # Create figure
        fig = bokeh.plotting.figure(
            width = self.width,
            height = self.height,
            x_range = self.x_range,
            y_range = self.y_range,
            x_axis_label='x [px]',
            y_axis_label='y [px]',
            active_scroll='wheel_zoom',
            name = name,
            )

        # Pass on x_range to other figures for same FOV
        self.x_range = fig.x_range
        self.y_range = fig.y_range
        
        return fig

#%%
    def figures_add_rgb(self, perc:float = 0.5):
        
        # Add figure
        fig = self._figure_create('fig_rgb')

        # Add rgb channel to source
        self._source_add_rgb(perc)

        # Add rgb image to figure
        img = fig.image_rgba(
            source=self.source,
            image='rgb',
            x='x',
            y='y',
            dw='width',
            dh='height',
            name='image_rgb',
        )

        # Add hover tool to rgb image
        hover_tool = bokeh.models.HoverTool(
            tooltips=[
                ('(x, y)', '($x{0.1a}, $y{0.1a})'),
            ],
            renderers = [img],
            description = 'Hover: RGB',
            name = 'hover_channel_rgb',
        )
        fig.add_tools(hover_tool)

        # Update
        self.figures['rgb'] = [fig]

        pass
        

#%%
    def figures_add_gray(self, channel_name: str, perc: float = 0.25):
        
        # Add figure
        fig = self._figure_create(f"fig_{channel_name}")
        
        # Add channel to source
        self._source_add_channel(channel_name)

        # Add colormapper
        vmin = np.nanpercentile(self.source.data[channel_name][0], perc)
        vmax = np.nanpercentile(self.source.data[channel_name][0], 100 - perc)
        color_mapper = bokeh.transform.LinearColorMapper(palette=self.palette , low=vmin , high=vmax, nan_color=self.nan_color)
        
        # Add gray scale image to figure
        img = fig.image(
            source=self.source,
            image=channel_name,
            x='x',
            y='y',
            dw='width',
            dh='height',
            color_mapper=color_mapper,
            name = f"image_{channel_name}",
        )

        # Add hover tool
        hover_tool = bokeh.models.HoverTool(
            tooltips=[
                    ('(x, y)', '($x{0.1a}, $y{0.1a})'),
                    (channel_name,f"@{channel_name}"),
            ],
            renderers = [img],
            description = f"Hover: {channel_name.upper()}",
            name = f"hover_channel_{channel_name}",
        )
        fig.add_tools(hover_tool)

        # Add  range slider and link to color_mapper range
        slider = bokeh.models.RangeSlider(
            start=vmin,
            end=vmax,
            value=(vmin,vmax),
            step=(vmax-vmin)/50,
            title=f"{channel_name} range:",
            width=self.width // 2,
            name = f"rangeslider_{channel_name}",
        )
        slider.js_link('value',color_mapper,'low', attr_selector=0)
        slider.js_link('value',color_mapper,'high', attr_selector=1)
        
        # Update
        self.figures[channel_name] = [fig, slider]
        
        pass


#%%
    def togglers_add_crowns(
        self,
        crowns: Dict,
        params_crowns: Dict,
        ) -> None:
        
        # Define name
        name = params_crowns['origin']

        # Update
        self._source_crowns_add(crowns, params_crowns)

        # Create toggler
        toggler = bokeh.models.Toggle(
                    active = True, 
                    label = "Show crowns:" + name,
                    button_type = "success",
                    width = self.width // 4,
                    height = self.height // 8,
                    )

        # Update
        self.togglers[name] = toggler

        # Add patches to all channels
        for key in self.figures:
            fig = self.figures[key][0]
            patches = fig.patches(
                source = self.source_crowns[name],
                xs = 'xs',
                ys = 'ys',
                line_color = 'black',
                color = 'white',
                alpha=self.patches_alpha,
                line_width = 1,
                visible = True,
                name = f"patches_{name}",
                )

            # Define hover tool tooltips
            tooltips = [(key, f"@{key}") for key in self.source_crowns[name].data.keys() if key not in ['xs', 'ys']]

            # Add hover tool
            hover_tool = bokeh.models.HoverTool(
                tooltips=tooltips,
                renderers = [patches],
                description = f"Hover: {name}",
                name = f"hover_patches_{name}",
            )
            fig.add_tools(hover_tool)
            
            # Link visibility of patches to toggler
            toggler.js_link('active', patches, 'visible')

            # Update
            self.figures[key][0] = fig

        pass

#%%
    def create_layout(self) -> bokeh.models.Column:
        
        cols = []

        for key in self.figures:
            
            # Define title division
            div = bokeh.models.Div(
                text=f"{key}".upper(),
                style={'font-size': self.title_font_size},
                width=self.width,
                height=self.height // 16,
                name = f"div_{key}",
                )

            # Define regular figure column as [div, fig, slider, ...]
            col_elements = [div]
            for item in self.figures[key]:
                col_elements.extend([item])

            # Create regular figure column and add to list of all columns  
            col = bokeh.layouts.column(col_elements)
            cols.extend([col])

        # Add all togglers to last column
        if len(self.togglers) > 0:
            col_last = bokeh.layouts.column([toggler for toggler in self.togglers.values()])
            cols.extend([col_last])
                
        layout = bokeh.layouts.layout([cols])
        
        return layout 

#####################################################################
#####################################################################


''' Use that piece of code maybe later, if it truns out that plotting of crowns as patches maybe too slow
'''
    # def togglers_add_bounds(self) -> None:
        
    #     # Assert that bounds are in source
    #     assert 'bounds' in self.source.data.keys(), "`bounds` not in self.source.data"
            
    #     # Add toggler for showing bounds
    #     toggler = bokeh.models.Toggle(
    #         active = True, 
    #         label = "Show crowns: watershed",
    #         button_type = "success",
    #         width = self.width // 4,
    #         height = self.height // 8,
    #         )
        
    #     # Loop through all figures and add bounds image connected to toggler
    #     for key in self.figures:
    #         fig = self.figures[key][0]

    #         # Add bounds
    #         bounds = fig.image(
    #             source=self.source,
    #             image='bounds',
    #             x='x',
    #             y='y',
    #             dw='width',
    #             dh='height',
    #             color_mapper=self.bounds_color_mapper,
    #             visible=True,
    #         )
    #         toggler.js_link('active', bounds, 'visible')

    #     # Update
    #     self.togglers['bounds'] = toggler

    #     pass