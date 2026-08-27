# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:39:01 2024

@main authors: Heng Zhang (heng.zhang@eawag.ch; hengzhang.zhh@gmail.com) & Laibao Liu (laibao@hku.hk)
@Swiss Federal Institute of Aquatic Science and Technology (EAWAG / ETH Domain) & The University of Hong Kong (HKU)
@Please cite the paper when using any part of this code/project. 

"""

import os
import numpy as np
from osgeo import gdal,osr

import src.processgeotiff4 as ptf

def getFileList(root,postfix):
    filelist=[]
    for root, dirs, files in os.walk(root):
        for file in files:
            if file.endswith(postfix):                
                filelist.append(os.path.join(root, file))
    Total=len(filelist)
    print("FileList Got!  Total = %d  " % Total)
    return filelist,Total

#%%
if __name__=="__main__":
    root=r"XXX"
    seasonrootfolderdir=r"XXX"
    modelattrfolderdir=root+os.sep+r"data\GCM_ModelAttr"
    PDrootfolderdir=root+os.sep+r"result\raster\wind_solar_opt_PD"
    resultrootfolderdir=root+os.sep+r"result\raster\wind_solar_season" 
    
    #model parameters
    modelNames=["ACCESS-ESM1-5","BCC-CSM2-MR","CanESM5","CESM2","CMCC-ESM2","EC-Earth3","GFDL-ESM4",\
                "IPSL-CM6A-LR","MPI-ESM1-2-HR","MRI-ESM2-0","NorESM2-MM","UKESM1-0-LL"]
    num_models=len(modelNames)
    
    # current time parameters
    # periods=["historical_1950_2014"]  
    # num_periods=len(periods)
    # cvYearStarts=[1980]
    # cvYearEnds=[2009]
    
    # future time parameters
    periods=["ssp245","ssp370","ssp585"]  
    # periods=["ssp585"]  
    num_periods=len(periods)
    cvYearStarts=[2036,2066]
    cvYearEnds=[2065,2095]

    moving_window_season_cycle=30   #unit: num. of days
    
#%%        
    for i_m in range(num_models):
        modelName=modelNames[i_m]
        for i_p in range(num_periods):
            period=periods[i_p]
            for i_y in range(len(cvYearStarts)):
                year_start=cvYearStarts[i_y]
                year_end=cvYearEnds[i_y]            
                print("processing...   period: %s, model: %s, time period: %d - %d "%(period,modelName,year_start,year_end))                

                #read the seasonal cycle numpy files
                seasonfolderdir=seasonrootfolderdir+os.sep+period+os.sep+modelName    
                wsfilename="wind_season_array_"+str(year_start)+"_"+str(year_end)+".npy"
                ssfilename="solar_season_array_"+str(year_start)+"_"+str(year_end)+".npy"
                wsfiledir=seasonfolderdir+os.sep+wsfilename
                ssfiledir=seasonfolderdir+os.sep+ssfilename
                wind_season_ori=np.load(wsfiledir)
                solar_season_ori=np.load(ssfiledir)
                w_array_shape=wind_season_ori.shape
                num_days_of_year=w_array_shape[0]
                num_lat=w_array_shape[1]
                num_lon=w_array_shape[2]
                
                #read power mean / density raster layers from the previou calculation
                windPDfolderdir=PDrootfolderdir+os.sep+"wind_no_season"+os.sep+period+os.sep+modelName  
                windPDfilename="wind_opt_pd_"+str(year_start)+"_"+str(year_end)+".tif"
                [wind_PD,driver,geoTransform,proj,nrow_PD,ncol_PD]=ptf.readSingleTiffAsNumpy(windPDfolderdir+os.sep+windPDfilename,datatype="Float32")
                solarPDfolderdir=PDrootfolderdir+os.sep+"solar_no_season"+os.sep+period+os.sep+modelName  
                solarPDfilename="solar_opt_pd_"+str(year_start)+"_"+str(year_end)+".tif"              
                [solar_PD,driver,geoTransform,proj,nrow_PD,ncol_PD]=ptf.readSingleTiffAsNumpy(solarPDfolderdir+os.sep+solarPDfilename,datatype="Float32")
                
#%%
                #calculate inter-seasonal CV 
                wind_season_cv=np.std(wind_season_ori,axis=0)/(wind_PD+1e-6)
                solar_season_cv=np.std(solar_season_ori,axis=0)/(solar_PD+1e-6)
                
    #%%            
                #the original result is flipped
                wind_season_cv=np.flipud(wind_season_cv)
                solar_season_cv=np.flipud(solar_season_cv)
                
                wind_season_cv_sft=np.zeros([num_lat,num_lon],dtype=np.float32)
                solar_season_cv_sft=np.zeros([num_lat,num_lon],dtype=np.float32)               
                
                
                wind_season_cv_sft[:,0:int(num_lon/2)]=wind_season_cv[:,int(num_lon/2):num_lon]
                wind_season_cv_sft[:,int(num_lon/2):num_lon]=wind_season_cv[:,0:int(num_lon/2)]
                solar_season_cv_sft[:,0:int(num_lon/2)]=solar_season_cv[:,int(num_lon/2):num_lon]
                solar_season_cv_sft[:,int(num_lon/2):num_lon]=solar_season_cv[:,0:int(num_lon/2)]                

                #write results
                driver=gdal.GetDriverByName("GTiff")
                proj = osr.SpatialReference()
                proj.ImportFromEPSG(4326)
                proj = proj.ExportToWkt()
                geotrans=[-180,360/num_lon,0,90,0,-1*180/num_lat]
                
                dirto=resultrootfolderdir+os.sep+period+os.sep+modelName
                if not os.path.exists(dirto):
                    os.makedirs(dirto)
                    
                rasfilename1="wind_season_cv_"+str(year_start)+"_"+str(year_end)+".tif"
                rasfilename2="solar_season_cv_"+str(year_start)+"_"+str(year_end)+".tif"                
                       
                rasfiledir1=dirto+os.sep+rasfilename1  
                rasfiledir2=dirto+os.sep+rasfilename2  
                         
                ptf.writeNumpyToTiff(wind_season_cv_sft,driver,geotrans,proj,num_lat,num_lon,-9999,rasfiledir1,datatype='Float32')
                ptf.writeNumpyToTiff(solar_season_cv_sft,driver,geotrans,proj,num_lat,num_lon,-9999,rasfiledir2,datatype='Float32')                
                
                print("done.\n")
    print("ALL DONE.\n")
