import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import findatree.io as io
import findatree.transform as transform
import findatree.detect as detect

############################################# Define path to raw data
dir_names=[]
dir_names.extend([r'C:\Users\flori\Documents\mpi\repos\findatree\example-data']*3)

file_names=[]
file_names.extend(['DSM.tif'])
file_names.extend(['DGM.tif'])
file_names.extend(['coordinates.csv'])

paths=[os.path.join(dir_names[i],file_name) for i, file_name in enumerate(file_names)]

#%%
### Compute canopy height model (CHM)
data, gt, proj = transform.compute_chm(paths[0],paths[1])

#%%
### Detect local maxima within box
box = int(np.floor(2/0.095))
x,y,ng = detect.identify_in_image(data[:,:,2], box)

### Get height @ local_maxima
h = data[x,y,2]

#%%
### Load coordinates csv
cs = pd.read_csv(paths[-1])

### Convert geo-coordinates to pixel-coordinates base son gt&proj
c_x = cs.x_utm.values
c_y = cs.y_utm.values
c_x = (c_x - gt[0])/gt[1] #- 15
c_y = (c_y - gt[3])/gt[5] #+ 15

############################################### Plot
### Set lowest height considered
lowest_h = 10
positives = h > lowest_h

### Height histogram
f = plt.figure(0,figsize = [4,3])
f.clear()
f.subplots_adjust(bottom=0.2,left=0.2)
ax = f.add_subplot(111)
ax.hist(h,bins=np.linspace(-5,50))
ax.axvline(lowest_h,ls='--',c='k')
ax.set_xlabel('Height (m)')
ax.set_ylabel('Occurences')


### CHM with found 
f = plt.figure(1,figsize = [5,5])
f.clear()
ax = f.add_subplot(111)
mapp = ax.imshow(data[:,:,2],vmin=10,vmax=25,cmap='Greys')
plt.colorbar(mapp,ax=ax)
ax.scatter(y[positives],x[positives],s=5,c='r')

### Group 1
# ax.scatter(c_x -15,c_y +15,s=20,c='none',edgecolors='orange')
# ax.set_xlim(900,1100)
# ax.set_ylim(3000,2780)

### Group 2
# ax.scatter(c_x -15,c_y +15,s=20,c='none',edgecolors='orange')
# ax.set_xlim(900,1100)
# ax.set_ylim(3000,2780)

### All
# ax.scatter(c_x,c_y,s=20,c='none',edgecolors='orange')
# ax.set_xlim(0,np.shape(data)[1])
# ax.set_ylim(np.shape(data)[0],0)

