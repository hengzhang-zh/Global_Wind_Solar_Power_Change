# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:56:50 2019

@author: ZH
"""

import os
import copy
import numpy as np
from osgeo import gdal,osr

#convert coordinates (x,y) to indices (idx,idy) of data matrix
def world2Pixel(geoMatrix,x,y):
    ulX=geoMatrix[0]                 #lon (X) coordinate of the upper-left corner 
    ulY=geoMatrix[3]                 #lat (Y) coordinate of the upper-left corner
    xDist=geoMatrix[1]               #distance of each pixel at longitudinal direction
    yDist=geoMatrix[5]               #distance of each pixel at latitudinal direction
#    rtnX=geoMatrix[2]
#    rtnY=geoMatrix[4]
    idx=int((x-ulX)/xDist)
    idy=int((y-ulY)/yDist)  
    return (idx, idy)    

def readTiffAsNumpy(TiffList,datatype='Float32'):
#    print("Reading GeoTiff files...")
    total=len(TiffList)
    tmpfiledir=TiffList[0]
    tmp=gdal.Open(tmpfiledir)
    ncol=tmp.RasterXSize
    nrow=tmp.RasterYSize
    Driver=tmp.GetDriver()
    GeoTransform=tmp.GetGeoTransform()
    Proj=tmp.GetProjection()
    if datatype=="Int8":
        dtp=np.int8
    elif datatype=="UInt8":
        dtp=np.uint8
    elif datatype=="Int16":
        dtp=np.int16
    elif datatype=="UInt16":
        dtp=np.uint16
    elif datatype=="Int32":
        dtp=np.int32
    elif datatype=="UInt32":
        dtp=np.uint32
    elif datatype=="Float32":
        dtp=np.float32
    elif datatype=="Float64":
        dtp=np.float64
    else:
        print("Data type not listed! Please choose from the bellowing:")
        print("Int8 UInt8 Int16 UInt16 UInt16 Int32 UInt32 Float32 Float64")    
    OriData=np.zeros([nrow,ncol,total],dtype=dtp)
    for i in range(total):
#        print("reading: %s"%TiffList[i])
        data=gdal.Open(TiffList[i])
        raster=data.ReadAsArray().astype(dtp)
        try:
            OriData[:,:,i]=raster
        except:
            OriData[:,:,i]=np.zeros([nrow,ncol],dtype=dtp)
    GeoTransform=np.array(GeoTransform)
    return [OriData,Driver,GeoTransform,Proj,nrow,ncol]

def readTiffLayersAsNumpy(TiffList, TiffLyrs, datatype='Float32'):
    # Determine basic dimensions from the first file
    tmpfiledir = TiffList[0]
    tmp = gdal.Open(tmpfiledir)
    ncol = tmp.RasterXSize
    nrow = tmp.RasterYSize
    Driver = tmp.GetDriver()
    GeoTransform = tmp.GetGeoTransform()
    Proj = tmp.GetProjection()
    
    # Streamlined data type mapping using a dictionary
    dtype_map = {
        "Int8": np.int8, "UInt8": np.uint8,
        "Int16": np.int16, "UInt16": np.uint16,
        "Int32": np.int32, "UInt32": np.uint32,
        "Float32": np.float32, "Float64": np.float64
    }
    
    if datatype in dtype_map:
        dtp = dtype_map[datatype]
    else:
        print("Data type not listed! Defaulting to Float32. Please choose from:")
        print("Int8 UInt8 Int16 UInt16 Int32 UInt32 Float32 Float64")  
        dtp = np.float32

    # Calculate total bands to extract across all files
    num_files = len(TiffList)
    num_lyrs = len(TiffLyrs)
    total_output_bands = num_files * num_lyrs

    # Initialize the 3D array: [rows, columns, total_extracted_bands]
    OriData = np.zeros([nrow, ncol, total_output_bands], dtype=dtp)
    
    for i in range(num_files):
        data = gdal.Open(TiffList[i])        
        if data is None:
            print(f"Warning: Could not open {TiffList[i]}")
            continue
        for j, lyr_idx in enumerate(TiffLyrs):
            # Calculate the corresponding Z-index in the final 3D array
            out_idx = (i * num_lyrs) + j            
            try:
                # GDAL is 1-indexed, so we add 1 to your 0-indexed TiffLyrs
                band = data.GetRasterBand(lyr_idx + 1)
                raster = band.ReadAsArray().astype(dtp)                
                if raster is not None:
                    OriData[:, :, out_idx] = raster
                else:
                    OriData[:, :, out_idx] = np.zeros([nrow, ncol], dtype=dtp)                    
            except Exception as e:
                print(f"Error reading layer {lyr_idx} in file {TiffList[i]}: {e}")
                OriData[:, :, out_idx] = np.zeros([nrow, ncol], dtype=dtp)                
    GeoTransform = np.array(GeoTransform)    
    return [OriData, Driver, GeoTransform, Proj, nrow, ncol]

def readSingleTiffAsNumpy(tiffFile,datatype='Float32'):
#    print("Reading GeoTiff files...")
    tmpfiledir=tiffFile
    tmp=gdal.Open(tmpfiledir)
    ncol=tmp.RasterXSize
    nrow=tmp.RasterYSize
    Driver=tmp.GetDriver()
    GeoTransform=tmp.GetGeoTransform()
    Proj=tmp.GetProjection()
    if datatype=="Int8":
        dtp=np.int8
    elif datatype=="UInt8":
        dtp=np.uint8
    elif datatype=="Int16":
        dtp=np.int16
    elif datatype=="UInt16":
        dtp=np.uint16
    elif datatype=="Int32":
        dtp=np.int32
    elif datatype=="UInt32":
        dtp=np.uint32
    elif datatype=="Float32":
        dtp=np.float32
    elif datatype=="Float64":
        dtp=np.float64
    else:
        print("Data type not listed! Please choose from the bellowing:")
        print("Int8 UInt8 Int16 UInt16 UInt16 Int32 UInt32 Float32 Float64")    

    data=gdal.Open(tiffFile)
    raster=data.ReadAsArray().astype(dtp)
    try:
        OriData=raster
    except:
        OriData=np.zeros([nrow,ncol],dtype=dtp)
    GeoTransform=np.array(GeoTransform)
    return [OriData,Driver,GeoTransform,Proj,nrow,ncol]

def getBlockRasterExtent(filelist):
    ras_bbox=np.array([9999999999,-9999999999,9999999999,-9999999999],dtype=np.float32)
    for i in range(len(filelist)):
        blockmapfiledir=filelist[i]
        tmp=gdal.Open(blockmapfiledir)
        geoTrans_tmp=tmp.GetGeoTransform()
        minX = geoTrans_tmp[0]
        maxY = geoTrans_tmp[3]
        maxX = minX + geoTrans_tmp[1] * tmp.RasterXSize
        minY = maxY + geoTrans_tmp[5] * tmp.RasterYSize
        ras_bbox[0]=min(ras_bbox[0],minX)
        ras_bbox[1]=max(ras_bbox[1],maxX)
        ras_bbox[2]=min(ras_bbox[2],minY)
        ras_bbox[3]=max(ras_bbox[3],maxY)
        del tmp
    geoTrans=np.array(geoTrans_tmp)
    geoTrans[0]=ras_bbox[0]
    geoTrans[3]=ras_bbox[3]
    ncol=int(0.5+(ras_bbox[1]-ras_bbox[0])/abs(geoTrans[1]))
    nrow=int(0.5+(ras_bbox[3]-ras_bbox[2])/abs(geoTrans[5]))
    return [geoTrans,ras_bbox,nrow,ncol]

def readImageBlocksAsNumpy(rasfolderdir,rasfilenames,datatype='Float32'):
    if datatype=="Int8":
        dtp=np.int8
    elif datatype=="UInt8":
        dtp=np.uint8
    elif datatype=="Int16":
        dtp=np.int16
    elif datatype=="UInt16":
        dtp=np.uint16
    elif datatype=="Int32":
        dtp=np.int32
    elif datatype=="UInt32":
        dtp=np.uint32
    elif datatype=="Float32":
        dtp=np.float32
    elif datatype=="Float64":
        dtp=np.float64
    else:
        print("Data type not listed! Please choose from the bellowing:")
        print("Int8 UInt8 Int16 UInt16 UInt16 Int32 UInt32 Float32 Float64")    
    
    # step 1: to generate raster file directions
    rasfiledirs=copy.deepcopy(rasfilenames)
    num_row_files=len(rasfilenames)
    num_col_files=len(rasfilenames[0])
    for j in range(num_row_files):
        for i in range(num_col_files):
            rasfiledirs[j][i]=rasfolderdir+os.sep+rasfilenames[j][i]
    
    #step 2: to calculate the dimensions of the large raster
    nrow,ncol=0,0
    for j in range(num_row_files):
        tmp=gdal.Open(rasfiledirs[j][0])
        nrow_tmp=tmp.RasterYSize
        nrow+=nrow_tmp
    for i in range(num_col_files):
        tmp=gdal.Open(rasfiledirs[0][i])
        ncol_tmp=tmp.RasterXSize
        ncol+=ncol_tmp        
    
    #step 3: to read the raster files
    ras_layer=np.zeros([nrow,ncol],dtype=dtp)
    i_row_start=0
    for j in range(num_row_files):
        i_col_start=0
        for i in range(num_col_files):
            print("reading image block: (%d, %d) / (%d, %d)"%(j+1,i+1,num_row_files,num_col_files))
            [mapLayers,Driver_block,GeoTransform_block,Proj_block,nrow_block,ncol_block]=readTiffAsNumpy([rasfiledirs[j][i]],datatype=datatype)
            mapLayers=mapLayers[:,:,0]
            i_row_end=i_row_start+nrow_block
            i_col_end=i_col_start+ncol_block
            ras_layer[i_row_start:i_row_end,i_col_start:i_col_end]=mapLayers
            # print([i_row_start,i_row_end,i_col_start,i_col_end])
            i_col_start+=ncol_block
        i_row_start+=nrow_block
    
    #step 4: to read the top-left block geoinfo
    tmp=gdal.Open(rasfiledirs[0][0])
    Driver=tmp.GetDriver()
    GeoTransform=tmp.GetGeoTransform()
    Proj=tmp.GetProjection()
    return [ras_layer,Driver,GeoTransform,Proj,nrow,ncol]

def readGEEImgBand(BN_name,imageBlockName_postfixes,imgfolderdir,datatype='Float32'):
    BN_BNs=copy.deepcopy(imageBlockName_postfixes)
    num_row_files=len(BN_BNs)
    num_col_files=len(BN_BNs[0])
    for j in range(num_row_files):
        for i in range(num_col_files):
            BN_BNs[j][i]=BN_name+imageBlockName_postfixes[j][i]  
    return readImageBlocksAsNumpy(imgfolderdir,BN_BNs,datatype=datatype)

def clipRas2Rect(ras,geoTrans):
    cd_row_sum=np.sum(ras,axis=1)
    cd_col_sum=np.sum(ras,axis=0)
    row_start=np.where(cd_row_sum>0)[0][0]
    row_end=np.where(cd_row_sum>0)[0][-1]+1
    col_start=np.where(cd_col_sum>0)[0][0]
    col_end=np.where(cd_col_sum>0)[0][-1]+1
    
    ras=np.ascontiguousarray(ras[row_start:row_end,col_start:col_end])
    
    geoTrans=list(geoTrans)
    geoTrans[0]=geoTrans[0]+geoTrans[1]*col_start
    geoTrans[3]=geoTrans[3]+geoTrans[5]*row_start
    
    nrow=int(row_end-row_start)
    ncol=int(col_end-col_start)  
    return [ras,geoTrans,nrow,ncol]

def calcRowColIndexBBOX(ras_shape,geoTrans,ras_bbox):
    row_start=max(0,int(0.5+(ras_bbox[3]-geoTrans[3])/geoTrans[5]))
    row_end=min(ras_shape[0],int(0.5+(ras_bbox[2]-geoTrans[3])/geoTrans[5]))
    col_start=max(0,int(0.5+(ras_bbox[0]-geoTrans[0])/geoTrans[1]))
    col_end=min(ras_shape[1],int(0.5+(ras_bbox[1]-geoTrans[0])/geoTrans[1]))
    if row_end==row_start:
        row_end+=1
    if col_end==col_start:
        col_end+=1    
    return [row_start,row_end,col_start,col_end]    

def clipRasBBOX(ras,geoTrans,ras_bbox):
    ras_shape=ras.shape
    [row_start,row_end,col_start,col_end]=calcRowColIndexBBOX(ras_shape,geoTrans,ras_bbox)    
    ras_clip=np.ascontiguousarray(ras[row_start:row_end,col_start:col_end])
    
    geoTrans_clip=list(copy.deepcopy(geoTrans))
    geoTrans_clip[0]=geoTrans[0]+geoTrans[1]*col_start
    geoTrans_clip[3]=geoTrans[3]+geoTrans[5]*row_start
    
    nrow=int(row_end-row_start)
    ncol=int(col_end-col_start)  
    return [ras_clip,geoTrans_clip,nrow,ncol]

def resizeImage(srcImage,dst_shape):
    img=Image.fromarray(srcImage)
    dstImage=np.array(img.resize(dst_shape,Image.Resampling.NEAREST),dtype=np.float32)
    return dstImage

def createTIFFDriverFromEPSG(EPSG_ID):
    driver=gdal.GetDriverByName("GTiff")
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(EPSG_ID)   #CH1903/LV03
    proj = proj.ExportToWkt()
    return [driver,proj]

def writeNumpyToTiff(TargetData,Driver,GeoTransform,Proj,nrow,ncol,nanDefault,filedirto,datatype='Float32'):
    if datatype=='UInt8':
        output=Driver.Create(filedirto,ncol,nrow,1,gdal.GDT_Byte)
        TargetData=TargetData.astype(np.uint8)
    elif datatype=='Int16':        
        output=Driver.Create(filedirto,ncol,nrow,1,gdal.GDT_Int16)
        TargetData=TargetData.astype(np.int16)
    elif datatype=='Int32':
        output=Driver.Create(filedirto,ncol,nrow,1,gdal.GDT_Int32)
        TargetData=TargetData.astype(np.int32)  
    elif datatype=='UInt16':        
        output=Driver.Create(filedirto,ncol,nrow,1,gdal.GDT_UInt16)
        TargetData=TargetData.astype(np.uint16)
    elif datatype=='UInt32':
        output=Driver.Create(filedirto,ncol,nrow,1,gdal.GDT_UInt32)
        TargetData=TargetData.astype(np.uint32)  
    elif datatype=='Float32':        
        output=Driver.Create(filedirto,ncol,nrow,1,gdal.GDT_Float32)
        TargetData=TargetData.astype(np.float32)
    elif datatype=='Float64':
        output=Driver.Create(filedirto,ncol,nrow,1,gdal.GDT_Float64)
        TargetData=TargetData.astype(np.float64)        
    else:
        print("Data type not listed! Please choose from the bellowing:")
        print("UInt8  Int16  Int32  UInt16  UInt32  Float32  Float64")
    output.SetGeoTransform(GeoTransform)
    output.SetProjection(Proj)
    outBand=output.GetRasterBand(1)
#    outBand.SetNoDataValue(nanDefault)    
    outBand.WriteArray(TargetData,0,0)
    outBand.FlushCache()
    
