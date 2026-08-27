# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 16:24:51 2023

@main authors: Heng Zhang (heng.zhang@eawag.ch; hengzhang.zhh@gmail.com) & Laibao Liu (laibao@hku.hk)
@Swiss Federal Institute of Aquatic Science and Technology (EAWAG / ETH Domain) & The University of Hong Kong (HKU)
@Please cite the paper when using any part of this code/project. 

"""
import os
import numpy as np
import scipy

import src.init as init
import src.processgeotiff4 as ptf

#generate destined points (water may flow through) based on given sampling points
def genDstPoints(point_coord,GeoTransform,radius_dist=100):
    [pX, pY] = ptf.world2Pixel(GeoTransform, point_coord[0], point_coord[1])  
    p_size=max(abs(GeoTransform[1]),abs(GeoTransform[5]))
    radius_num=int(radius_dist/p_size)
    dst_points=[]
    for j in range(pY-radius_num,pY+radius_num+1):
        for i in range(pX-radius_num,pX+radius_num+1):
            dist=np.sqrt(np.square(j-pY)+np.square(i-pX))
            if dist<=radius_num:
                dst_points.append([j,i])
    return dst_points

#initialize catchment mask and catchment distance data
def initCatchDist(dst_points,nrow,ncol,default_dist_value):
    catchdist=np.zeros([nrow,ncol],dtype=np.float32)
    for dp in dst_points:
        catchdist[dp[0],dp[1]]=default_dist_value   #here -1 stands for the outlet of catchment
    return catchdist

#%%
if __name__=="__main__":
    root=r"XXX"
    rasfolderdir=r"XXX"
    csvfolderdir=root+os.sep+r"data\global_power_plant"
    resultfolderdir=root+os.sep+r"result\table\global_solar_wind_power_plant_stats"

    lon_header="Longitude"
    lat_header="Latitude"
    
    radius_pix=1   #the radius in pixel to extract circle buffer
    filter_method="mode"
    
    ras_header_prefix="sign_change_"
    
    csvfile_strs=["wind_L_scale", "wind_under_thres", "wind_all_scale",\
                  "solar_L_scale", "solar_M_scale", "solar_under_thres", "solar_all_scale"]
    for csvfile_str in csvfile_strs:  
        csvfilename="global_power_tracker_2023_"+csvfile_str+".csv"
        csvfiledir=csvfolderdir+os.sep+csvfilename
        pwplPD=init.readCSVasPandas(csvfiledir)        
        num_sites=pwplPD.shape[0]

        [rasfile_dirs,num_ras]=init.getFileList(rasfolderdir, ".tif")
        
        if ras_header_prefix=="rate_change_":
            site_ras_values=np.zeros([num_sites,num_ras+1],dtype=np.float32)
        else:
            site_ras_values=np.zeros([num_sites,num_ras+1],dtype=np.uint16)
        site_ras_headers=["siteID"]
    #%%
        for i_ras in range(num_ras):
            rasfile_dir=rasfile_dirs[i_ras]
            rasfile_str=rasfile_dir.split(os.sep)[-1].split(".tif")[0]
            site_ras_header=rasfile_str.split(ras_header_prefix)[1].split("_historical")[0]
            site_ras_headers.append(site_ras_header)
            
            rasfilename=rasfile_str+".tif"
            rasfiledir=rasfolderdir+os.sep+rasfilename
            [raster,driver,geoTrans,proj,nrow,ncol]=ptf.readSingleTiffAsNumpy(rasfiledir,datatype='Float32') 
            if ras_header_prefix=="rate_change_":
                nandefault=raster[0,0]
                raster[raster==nandefault]=np.nan
        
        #%%
            #iteration: to extract subset of one catchment
            for i_site in range(0,num_sites):
                rec_coords=np.array([pwplPD.loc[i_site,lon_header],pwplPD.loc[i_site,lat_header]])
                dst_points=genDstPoints(rec_coords,geoTrans,radius_dist=geoTrans[1]*radius_pix)
                site_ras_values[i_site,0]=i_site
                
                try:
                    num_points=len(dst_points)
                    pixel_values=np.zeros(num_points,dtype=np.float32)
                    for i_p in range(num_points):
                        pixel_values[i_p]=raster[dst_points[i_p][0],dst_points[i_p][1]]
                    pixel_values=pixel_values[np.isfinite(pixel_values)]
                    if filter_method=="mean":
                        site_ras_values[i_site,i_ras+1]=np.nanmean(pixel_values)
                    elif filter_method=="median":
                        site_ras_values[i_site,i_ras+1]=np.nanmedian(pixel_values)
                    elif filter_method=="max":
                        site_ras_values[i_site,i_ras+1]=np.nanmax(pixel_values)
                    elif filter_method=="min":
                        site_ras_values[i_site,i_ras+1]=np.nanmin(pixel_values)
                    elif filter_method=="mode":
                        if ras_header_prefix=="rate_change_":
                            pixel_values=pixel_values[pixel_values!=0]
                            site_ras_values[i_site,i_ras+1]=np.float32(scipy.stats.mode(pixel_values,axis=None)[0])
                        else:
                            site_ras_values[i_site,i_ras+1]=np.uint16(scipy.stats.mode(pixel_values,axis=None)[0])
                    else:
                        print("invalid input method.")
                    print("extracting values of %s in %s... \n site: (%d / %d), value = %.2f"%(csvfile_str,rasfile_str,i_site,num_sites,site_ras_values[i_site,i_ras+1]))     
                except:
                    print("unable to extract value...")
                    site_ras_values[i_site,i_ras+1]=np.nan
        
        #%%
        # write the results
        if not os.path.exists(resultfolderdir):
            os.makedirs(resultfolderdir)
        csvfilename1="ras_"+ras_header_prefix+"values_"+csvfile_str+".csv"
        csvfiledir1=resultfolderdir+os.sep+csvfilename1
        init.writeArrayToCSV(site_ras_values,site_ras_headers,csvfiledir1)     
        print("finish.")
    print("ALL DONE.\n")
    
