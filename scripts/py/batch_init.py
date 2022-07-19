#%%
import importlib
import os
from tqdm import tqdm

import findatree.io as io
import findatree.transformations as transformations
import findatree.interactive as interactive
import findatree.geo_to_image as geo_to_image
import findatree.segmentation as segmentation
import findatree.object_properties as object_properties

from bokeh.plotting import save
from bokeh.io import output_file

importlib.reload(io)
importlib.reload(transformations)
importlib.reload(interactive)
importlib.reload(geo_to_image)
importlib.reload(segmentation)
importlib.reload(object_properties)


#%%
# Define full paths to directories containing dsm, dtm, ortho and shape-files 
dir_names=[]
dir_names.extend([r'C:\Data\lwf\DSM_2020'])
dir_names.extend([r'C:\Data\lwf\DTM'])
dir_names.extend([r'C:\Data\lwf\Orthophotos_2020'])
dir_names.extend([r'C:\Data\lwf\CrownDelineation_2020'])

# Define directories where return hdf5s and htmls are stored
save_dir_hdf5 = r"C:\Data\lwf\processed\2020\hdf5"
save_dir_html = r"C:\Data\lwf\processed\2020\html"

# Get all available tnr numbers in directory
tnrs = io.find_all_tnrs_in_dir(dir_names[0])



#%%

tnr_exceptions = []

# Main batch loop to save channels & crowns in hdf5 and create preliminary bokeh plot html
for tnr in tqdm(tnrs):
    
    print()
    print(f"Processing: tnr_{tnr}")

    ################# [1] Load and save channels
    try:
        params_channels = {'tnr': tnr}
        channels, params_channels = geo_to_image.channels_load(dir_names, params_channels, verbose=False)
        io.channels_to_hdf5(channels, params_channels, dir_name = save_dir_hdf5)
    
    except:
        print(f"-> tnr_{tnr}: Exception during loading/saving of channels")
        tnr_exceptions.extend([tnr])

    ################# [2] Load and save human generated crowns
    try:
        crowns_human, params_crowns_human = io.load_shapefile(dir_names, params_channels, verbose=False)
        io.crowns_to_hdf5(crowns_human, params_crowns_human, dir_name = save_dir_hdf5)
    
    except:
        print(f"-> tnr_{tnr}: Exception during loading/saving of crowns_human")
        tnr_exceptions.extend([tnr])

    ################# [3] Generate and save watershed crowns
    try:
        params_crowns_water = {}

        crowns_water, params_crowns_water = segmentation.watershed(
            channels,
            params_channels,
            params_crowns_water,
            verbose=False,
        )
        io.crowns_to_hdf5(crowns_water, params_crowns_water, dir_name = save_dir_hdf5)
    
    except:
        print(f"-> tnr_{tnr}: Exception during generation/saving of crowns_water")
        tnr_exceptions.extend([tnr])

    ################# [4] Create bokeh plot and save as html
    try:
        # Init Plotter object and adjust some attributes 
        plt = interactive.Plotter()
        plt.width = 500
        plt.channels_downscale = 1

        # Add channels
        plt.add_channels(channels, params_channels)
        plt.figures_add_rgb()
        plt.figures_add_gray('chm')

        # Add crowns
        plt.togglers_add_crowns(crowns_water, params_crowns_water)
        plt.togglers_add_crowns(crowns_human, params_crowns_human)

        # Create and save layout
        layout = plt.create_layout()
        output_file(
            filename = os.path.join(save_dir_html, f"tnr{tnr}.html"),
            title = f"tnr{tnr}_{transformations.current_datetime()}",
            mode='inline',
            )
        save(layout)

    except:
        print(f"-> tnr_{tnr}: Exception during creation of html")
        tnr_exceptions.extend([tnr])


print()
print('Done!')
print(f"Exceptions occured @{tnr_exceptions}")