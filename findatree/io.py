from osgeo import gdal


NP2GDAL_DTYPES = {'uint8': 1,
                  'int8': 1,
                  'uint16': 2,
                  'int16': 3,
                  'uint32': 4,
                  'int32': 5,
                  'float32': 6,
                  'float64': 7,
                  'complex64': 10,
                  'complex128': 11,
                  }

def load_dsm(path):
    
    ### Open dsm (digital surface model )and dtm (digital terrain model)
    ds = gdal.Open(path,gdal.GA_ReadOnly)
    
    ### Get projection
    proj = ds.GetProjection()
    
    ### Get affine geotransform
    gt = ds.GetGeoTransform()
    
    ### Pixel extents
    shape = [ds.RasterXSize,ds.RasterYSize]
    
    ### Driver ShortName
    driver = ds.GetDriver().ShortName
    
    ### Number of RasterBands and dtypes
    rastercount = ds.RasterCount
    rasterdtype = [ds.GetRasterBand(i+1).DataType for i in range(rastercount)]
    
    ### Print some info
    print('    shape [px]: (%i,%i)'%(shape[0],shape[1]))
    print('    resolution [m]: (%.3f,%.3f)'%(gt[1],-gt[-1]))
    print('    number of rasters: %i'%rastercount)
    
    return ds, proj, gt, shape, rastercount, rasterdtype
    
    