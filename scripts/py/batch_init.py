#%%
import importlib

import findatree.io as io
import findatree.transformations as transformations
import findatree.interactive as interactive
import findatree.geo_to_image as geo_to_image
import findatree.segmentation as segmentation
import findatree.object_properties as object_properties

# from bokeh.plotting import save
# from bokeh.io import output_file

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

# Get all available tnr numbers in directory
tnrs = io.find_all_tnrs_in_dir(dir_names[0])
