import os
import numpy as np
import matplotlib.pyplot as plt

import findatree.io as io
import findatree.transform as transform
import findatree.detect as detect

############################################# Define path to raw data
dir_names=[]
dir_names.extend([r'C:\Users\flori\Documents\mpi\repos\findatree\example-data'])
dir_names.extend([r'C:\Users\flori\Documents\mpi\repos\findatree\example-data'])

file_names=[]
file_names.extend(['DSM.tif'])
file_names.extend(['DGM.tif'])

paths=[os.path.join(dir_names[i],file_name) for i, file_name in enumerate(file_names)]

#%%
### Compute canopy height model (CHM)
data = transform.compute_chm(paths[0],paths[1])

#%%
### Detect local maxima within box
box = int(np.floor(3/0.095))
x,y,ng = detect.identify_in_image(data[:,:,2], box)

### Get height @ local_maxima
h = data[x,y,2]

#%%
############################################### Plot
### Set lowest height considered
lowest_h = 14
positives = h > lowest_h

### Height histogram
f = plt.figure(0,figsize = [5,4])
f.clear()
ax = f.add_subplot(111)
ax.hist(h,bins=np.linspace(-5,50))
ax.axvline(lowest_h,ls='--',c='k')
ax.set_xlabel('Height (m)')
ax.set_ylabel('Occurences')


### CHM with found 
f = plt.figure(1,figsize = [6,6])
f.clear()
ax = f.add_subplot(111)
mapp = ax.imshow(data[:,:,2],vmin=0,vmax=30,cmap='Greys')
plt.colorbar(mapp,ax=ax)
ax.scatter(y[positives],x[positives],s=10,c='r')
