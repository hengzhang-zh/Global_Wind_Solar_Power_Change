# -*- coding: utf-8 -*-
"""
Created on Sat May 11 13:52:04 2024

@main authors: Heng Zhang (heng.zhang@eawag.ch; hengzhang.zhh@gmail.com) & Laibao Liu (laibao@hku.hk)
@Swiss Federal Institute of Aquatic Science and Technology (EAWAG / ETH Domain) & The University of Hong Kong (HKU)
@Please cite the paper when using any part of this code/project. 

"""

#required packages: gdal & pycuda
#please note that the catchment computing only runs on a NVIDIA GPU. 
#please translate the CUDA kernal code to ROCm/C++ if you would like to use an AMD GPU or CPUs

import os
import time
import numpy as np
import netCDF4 as nc

#CUDA environment
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

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
    windrootfolderdir=r"XXX"
    solarrootfolderdir=r"XXX" 
    modelattrfolderdir=root+os.sep+r"data\GCM_ModelAttr"
    resultrootfolderdir=r"F:\Global_Solar_Wind_Energy\GCM_Season_Cycle" 
    cufile=root+os.sep+r"code\Python\src\calcWSOPTRatioCV.cu"
    
    #model parameters
    modelNames=["ACCESS-ESM1-5","BCC-CSM2-MR","CanESM5","CESM2","CMCC-ESM2","EC-Earth3","GFDL-ESM4",\
                "IPSL-CM6A-LR","MPI-ESM1-2-HR","MRI-ESM2-0","NorESM2-MM","UKESM1-0-LL"]
    # modelNames=["CESM2"]
    num_models=len(modelNames)
    power_name="wind"
    
    BATCHSIZE=144
    bool_mask_land=False
    moving_window_years=10
    
    # #current time parameters
    # periods=["historical_1950_2014"]  
    # num_periods=len(periods)
    # tsYearStarts=[1980]
    # tsYearEnds=[2009]
    
    #future time parameters
    periods=["ssp245","ssp245","ssp370","ssp370","ssp585","ssp585"]  
    num_periods=len(periods)
    tsYearStarts=[2036,2066,2036,2066,2036,2066]
    tsYearEnds=[2065,2095,2065,2095,2065,2095]    
    
    #CUDA implementation
    #step 1: read cu file and compile the CUDA code
    with open(cufile,"rt") as f:
        cu_src = f.read()
    
    mod = SourceModule(cu_src)    
    func1 = mod.get_function("calcMovWinAvgTrend")
    func2 = mod.get_function("calcSeasonalEffect")
#%%
    for i_p in range(num_periods):
        period=periods[i_p]
        year_start=tsYearStarts[i_p]
        year_end=tsYearEnds[i_p]
        for i_m in range(num_models):
#%%
            modelName=modelNames[i_m]
            print("processing...   period: %s, model: %s, time period: %d - %d "%(period,modelName,year_start,year_end))
            
            dirto=resultrootfolderdir+os.sep+period+os.sep+modelName
            pvfilename=power_name+"_season_array_"+str(year_start)+"_"+str(year_end)+".npy"
            pvfiledir=dirto+os.sep+pvfilename
            if os.path.exists(pvfiledir):
                print("this file finished. continue...")
                continue              

            modelattrfiledir=modelattrfolderdir+os.sep+"Basic_"+modelName+".nc"
            if power_name=="wind":
                powerfolderdir=windrootfolderdir+os.sep+period+os.sep+modelName                
            elif power_name=="solar":
                powerfolderdir=solarrootfolderdir+os.sep+period+os.sep+modelName
            else:
                print("wrong input power type! ")

            powerfiledir,total=getFileList(powerfolderdir,".nc")
            powerfiledir=powerfiledir[0]
            
            #to extract the years of start and end
            s=powerfiledir.split(os.sep)[-1].split(".nc")[0]            
            nc_year_start_power=int(s[-17:-13])
            nc_year_end_power=int(s[-8:-4])
            
            powerncds = nc.Dataset(powerfiledir)
            modelattr = nc.Dataset(modelattrfiledir)
            
            power_dim=powerncds.dimensions
            num_lon=power_dim["lon"].size
            num_lat=power_dim["lat"].size
            num_time=power_dim["time"].size
            
            #to find targetted time series slice ids
            model_time=np.transpose(np.array(modelattr.variables["Yr-Mon-Day"][:]).astype(np.float32))
            model_time_year=model_time[:,0]
            year_start_mv=max(year_start-int(moving_window_years/2),nc_year_start_power)
            year_end_mv=min(year_end+moving_window_years-int(moving_window_years/2)-1,nc_year_end_power)
        
            nc_time=model_time[np.where((model_time_year>=nc_year_start_power)&(model_time_year<=nc_year_end_power))[0],:]
            ts_time_ids=np.where((nc_time[:,0]>=year_start_mv)&(nc_time[:,0]<=year_end_mv))[0]
            ts_leap_ids=np.where((nc_time[:,1]==2)&(nc_time[:,2]==29))[0]
            ts_time_ids = np.setdiff1d(ts_time_ids, ts_leap_ids)  #remove Feb 29 in leap years
            
            #to read the wind, solar power data and time series, there is a special case for "wind ssp585"
            if power_name=="wind":  
                if period=="ssp585":
                    power_pv=np.array(powerncds.variables["pv_wind150"][ts_time_ids]).astype(np.float32)                    
                else:
                    power_pv=np.array(powerncds.variables["pv_wind150"][:].take(ts_time_ids,axis=0)).astype(np.float32)
            elif power_name=="solar":
                power_pv=np.array(powerncds.variables["PV"][:].take(ts_time_ids,axis=0)).astype(np.float32)
            else:
                print("wrong input power type! ")
            powerncds.close()
            modelattr.close()

            pv_time=nc_time[ts_time_ids,:]
            num_days_of_year=min(365,np.sum(pv_time[:,0]==year_start_mv))
            num_ori_time=len(ts_time_ids)         
            
            mv_size=len(np.where((pv_time[:,0]>=year_start_mv)&(pv_time[:,0]<=(year_start_mv+moving_window_years-1)))[0])
            ts_time_ids=ts_time_ids[(int(mv_size/2)):(num_ori_time-int(mv_size/2))]
            num_time=num_ori_time-mv_size
            ts_mv_time=nc_time[ts_time_ids,:]
            
#%%
            power_pv_season=np.zeros([num_days_of_year,num_lat,num_lon],dtype=np.float32)
        
            #to divide image into batches
            num_batch_lat=int(np.ceil(num_lat/BATCHSIZE))
            num_batch_lon=int(np.ceil(num_lon/BATCHSIZE))               
            for i_batch_lat in range(num_batch_lat):
                for i_batch_lon in range(num_batch_lon):
                    #find the boundary of the batch
                    i_batch_y_start=i_batch_lat*BATCHSIZE
                    i_batch_y_end=min((i_batch_lat+1)*BATCHSIZE,num_lat)
                    i_batch_x_start=i_batch_lon*BATCHSIZE
                    i_batch_x_end=min((i_batch_lon+1)*BATCHSIZE,num_lon)
                    num_lat_batch=i_batch_y_end-i_batch_y_start
                    num_lon_batch=i_batch_x_end-i_batch_x_start
                    
                    power_pv_batch=np.ascontiguousarray(power_pv[:,i_batch_y_start:i_batch_y_end,i_batch_x_start:i_batch_x_end])
                    ws_opt_batch=gpuarray.zeros([num_lat_batch,num_lon_batch],dtype=np.float32)                    
                    power_pv_batch=gpuarray.to_gpu(power_pv_batch)
                    
#%%
                    #specify the block & grid sizes in CUDA kernel
                    BLOCKDIM=16
                    blockSize=(BLOCKDIM,BLOCKDIM,1)
                    bx=int((num_lon_batch+BLOCKDIM-1)/BLOCKDIM)
                    by=int((num_lat_batch+BLOCKDIM-1)/BLOCKDIM)
                    gridSize=(bx,by,1)     
                    
                    print("GPU computing using batch...        batch: (%d, %d) / (%d, %d)"%(i_batch_lat+1,i_batch_lon+1,num_batch_lat,num_batch_lon))
                    t1=time.time()
                    #step 1: remove trend using moving window average 
                    power_pv_batch_trend=gpuarray.zeros([num_time,num_lat_batch,num_lon_batch],dtype=np.float32)
                    power_pv_batch_detrd=gpuarray.zeros([num_time,num_lat_batch,num_lon_batch],dtype=np.float32)
                    power_pv_batch_trend_mean=gpuarray.zeros([num_lat_batch,num_lon_batch],dtype=np.float32)
                    
                    print("calculating moving window average trend...")
                    func1(power_pv_batch_trend,power_pv_batch_detrd,power_pv_batch_trend_mean,power_pv_batch,np.int32(mv_size),np.int32(num_ori_time), np.int32(num_lat_batch), np.int32(num_lon_batch),block=blockSize,grid=gridSize)
                    cuda.Context.synchronize()
                    del power_pv_batch_trend,power_pv_batch
    
                    t2=time.time()
                    print("computing finished.    time = %d s.  "%(t2-t1))                      
                    
                    #step 2: remove seasonal cycle on the daily basis
                    print("calculating seasonal cycle (on daily basis)...")
                    power_pv_batch_season=gpuarray.zeros([num_days_of_year,num_lat_batch,num_lon_batch],dtype=np.float32)  
                    power_pv_batch_dessn=gpuarray.zeros([num_time,num_lat_batch,num_lon_batch],dtype=np.float32)
                    
                    func2(power_pv_batch_season,power_pv_batch_dessn,power_pv_batch_detrd,np.int32(num_days_of_year),np.int32(num_time), np.int32(num_lat_batch), np.int32(num_lon_batch),block=blockSize,grid=gridSize)
                    cuda.Context.synchronize()
                    power_pv_season[:,i_batch_y_start:i_batch_y_end,i_batch_x_start:i_batch_x_end]=power_pv_batch_season.get()
                    del power_pv_batch_detrd, power_pv_batch_dessn, power_pv_batch_season
                    
                    t3=time.time()
                    print("computing finished.    time = %d s.  "%(t3-t2))   
                
#%%
            #save numpy arrays wind_season & solar_season to local directory
            dirto=resultrootfolderdir+os.sep+period+os.sep+modelName
            if not os.path.exists(dirto):
                os.makedirs(dirto)
            pvfilename=power_name+"_season_array_"+str(year_start)+"_"+str(year_end)+".npy"
            pvfiledir=dirto+os.sep+pvfilename
            np.save(pvfiledir,power_pv_season)
            print("done.\n")
    print("ALL DONE.\n")
