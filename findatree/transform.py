import os
from osgeo import gdal
import numpy as np

import findatree.io as io
#%%
def px_to_geo(xpx, ypx, gt):
    '''
    Convert pixel-coordinates ``xpx,ypx`` to geo-coordinates using gdal's affine geotransform ``gt``.
    '''
    xgeo = gt[0] + xpx*gt[1] + ypx*gt[2]
    ygeo = gt[3] + xpx*gt[4] + ypx*gt[5]
    return xgeo, ygeo

#%%
def get_geo_extent(gt,shape):
    '''
    Get extent ``[xgeo[0],xgeo[0],ygeo[1],ygeo[1]]`` of DSM in geo-coordinates 
    using gdal's affine geotransform ``gt``` and shape in pixels ``shape``.
    '''
    extent = [gt[0],
              gt[3],
              px_to_geo(shape[0],0,gt)[0],
              px_to_geo(0,shape[1],gt)[1],
              ]
    return extent

#%%
def reproject_ds(ds,shape,dtype,gt,proj):
    '''
    Reproject gdal.Dataset to new projection and geotransfrom and return rasters as numpy.arrays.
    '''
    ### Init gdal.Dataset in memory only
    ds_reproject = gdal.GetDriverByName('MEM').Create('',
                                                      shape[0],
                                                      shape[1],
                                                      1,
                                                      dtype)
    ds_reproject.SetGeoTransform(gt)
    ds_reproject.SetProjection(proj)
    
    ### Reproject
    gdal.ReprojectImage(ds,
                        ds_reproject,
                        ds.GetProjection(),
                        proj,
                        gdal.GRA_Bilinear,#gdal.GRA_NearestNeighbour,
                        )
    ### Get raster data only
    ds_data = ds_reproject.ReadAsArray()
    
    ### Close gdal.Dataset
    ds_reproject = None
    
    return ds_data

#%%
def compute_chm(dsm_path, dtm_path):
    
    ### Open dsm (digital surface model )and dtm (digital terrain model)
    print()
    print(os.path.split(dsm_path)[-1])
    dsm_ds, dsm_proj, dsm_gt, dsm_shape, dsm_rastercount, dsm_rasterdtype = io.load_dsm(dsm_path)
    print()
    print(os.path.split(dtm_path)[-1])
    dtm_ds, dtm_proj, dtm_gt, dtm_shape, dtm_rastercount, dtm_rasterdtype = io.load_dsm(dtm_path)
    
    ### Set resolution: Minimum resolution in xy in both dsm and dtm in geo coordinates
    res = min(dsm_gt[1],-dsm_gt[-1],dtm_gt[1],-dtm_gt[-1])
    print()
    print('Resolution set to: %.3f'%res)
    
    ### Set output-extent: Overlap of dsm and dtm in geo coordinates [x0,y0,x1,y1]
    dsm_extent = get_geo_extent(dsm_gt,dsm_shape)
    dtm_extent = get_geo_extent(dtm_gt,dtm_shape)
    
    extent = [max(dsm_extent[0],dtm_extent[0]),
              min(dsm_extent[1],dtm_extent[1]),
              min(dsm_extent[2],dtm_extent[2]),
              max(dsm_extent[3],dtm_extent[3]),
              ] 
    
    ### Define geotransform, shape in pixels, projection and dtype of output
    gt = [extent[0],res,0,
          extent[1],0,-res]
    shape = (int(np.floor((extent[2] - extent[0]) / res)),
             int(np.floor((extent[1] -extent[3]) / res)),
             )
    proj = dsm_proj
    dtype = io.NP2GDAL_DTYPES['float32']
    
    ### Reproject DSM and DTM and return rasters
    dsm = reproject_ds(dsm_ds,shape,dtype,gt,proj)
    dtm = reproject_ds(dtm_ds,shape,dtype,gt,proj)
    
    ### Compute final rasters consiting of reprojected DSM [::0], reprojected DTM [::1] and CHM [::2]
    data = np.zeros([shape[1],shape[0],3])
    data[:,:,0] = dsm
    data[:,:,1] = dtm
    data[:,:,2] = data[:,:,0] - data[:,:,1]
    
    return data
