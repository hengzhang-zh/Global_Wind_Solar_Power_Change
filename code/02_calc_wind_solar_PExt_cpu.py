# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 15:42:16 2025

@main authors: Heng Zhang (heng.zhang@eawag.ch; hengzhang.zhh@gmail.com) & Laibao Liu (laibao@hku.hk)
@Swiss Federal Institute of Aquatic Science and Technology (EAWAG / ETH Domain) & The University of Hong Kong (HKU)
@Please cite the paper when using any part of this code/project. 

"""

import os
import skfmm
import numpy as np
import netCDF4 as nc
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
    windrootfolderdir=r"XXX"
    solarrootfolderdir=r"XXX" 
    modelattrfolderdir=root+os.sep+r"data\GCM_ModelAttr"
    resultrootfolderdir=root+os.sep+r"result\raster\wind_solar_TS_analysis"
    
    bool_mask_land=False
    moving_window_years=0

    thres_under_above="under_thres"             #under_thres    above_thres
    historical_thres_perc=0.10
    
    #model parameters
    modelNames=["ACCESS-ESM1-5","BCC-CSM2-MR","CanESM5","CESM2","CMCC-ESM2","EC-Earth3","GFDL-ESM4",\
                "IPSL-CM6A-LR","MPI-ESM1-2-HR","MRI-ESM2-0","NorESM2-MM","UKESM1-0-LL"]
    num_models=len(modelNames)
    power_name="wind"

    # # current time parameters
    # periods=["historical_1950_2014"]    #must run "historical" first for a thres map, then apply it to the "future"
    # num_periods=len(periods)
    # tsYearStarts=[1980]
    # tsYearEnds=[2009]
    
    #future time parameters
    periods=["ssp245","ssp245","ssp370","ssp370","ssp585","ssp585"]   
    num_periods=len(periods)
    tsYearStarts=[2036,2066,2036,2066,2036,2066]
    tsYearEnds=[2065,2095,2065,2095,2065,2095]    
    
      
#%%
    for i_p in range(num_periods):
        period=periods[i_p]
        year_start=tsYearStarts[i_p]
        year_end=tsYearEnds[i_p]
        for i_m in range(num_models):
#%%
            modelName=modelNames[i_m]
            print("processing...   period: %s, model: %s, time period: %d - %d "%(period,modelName,year_start,year_end))
            
            #check if finished
            if thres_under_above=="under_thres":
                rasfolderdirto=resultrootfolderdir+os.sep+power_name+"_incl_season"+os.sep+"under_thres_days"+os.sep+period+os.sep+modelName
                rasfilename=power_name+"_under_thres"+str(int(historical_thres_perc*100)).zfill(3)+"_"+str(year_start)+"_"+str(year_end)+"_ext.tif"                  
            else:
                rasfolderdirto=resultrootfolderdir+os.sep+power_name+"_incl_season"+os.sep+"above_thres_days"+os.sep+period+os.sep+modelName
                rasfilename=power_name+"_above_thres"+str(int(historical_thres_perc*100)).zfill(3)+"_"+str(year_start)+"_"+str(year_end)+"_ext.tif"
                 
            rasfiledir=rasfolderdirto+os.sep+rasfilename  
            if os.path.exists(rasfiledir):
                print("this file finished! continue. \n")
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
            #calculate low power threshold
            #please calculate the present first, then use the thres map for the future. 
            if period=="historical_1950_2014":
                if thres_under_above=="under_thres":
                    thresmapfolderdir=resultrootfolderdir+os.sep+power_name+"_incl_season"+os.sep+"under_thres_days"+os.sep+"historical_count_thres_map"
                    power_thres_map=np.percentile(power_pv,historical_thres_perc*100,axis=0)+0.1
                else:
                    thresmapfolderdir=resultrootfolderdir+os.sep+power_name+"_incl_season"+os.sep+"above_thres_days"+os.sep+"historical_count_thres_map"
                    power_thres_map=np.percentile(power_pv,historical_thres_perc*100,axis=0)-0.1
                
                driver=gdal.GetDriverByName("GTiff")
                proj = osr.SpatialReference()
                proj.ImportFromEPSG(4326)
                proj = proj.ExportToWkt()
                geotrans=[-180,360/num_lon,0,90,0,-1*180/num_lat]
                
                if not os.path.exists(thresmapfolderdir):
                    os.makedirs(thresmapfolderdir)                   
                thresmapfilename=power_name+"_historical_count_thres_map_"+modelName+"_thres"+str(int(historical_thres_perc*100)).zfill(3)+".tif"
                thresmapfiledir2=thresmapfolderdir+os.sep+thresmapfilename  
                ptf.writeNumpyToTiff(power_thres_map,driver,geotrans,proj,num_lat,num_lon,-9999,thresmapfiledir2,datatype='Float32')
            else:
                print("read existing count threshold map.")
                if thres_under_above=="under_thres":
                    thresmapfolderdir=resultrootfolderdir+os.sep+power_name+"_incl_season"+os.sep+"under_thres_days"+os.sep+"historical_count_thres_map"
                else:
                    thresmapfolderdir=resultrootfolderdir+os.sep+power_name+"_incl_season"+os.sep+"above_thres_days"+os.sep+"historical_count_thres_map"
                thresmapfilename=power_name+"_historical_count_thres_map_"+modelName+"_thres"+str(int(historical_thres_perc*100)).zfill(3)+".tif"
                thresmapfiledir2=thresmapfolderdir+os.sep+thresmapfilename      
                [power_thres_map,driver,geotrans,proj,nrow,ncol]=ptf.readTiffAsNumpy([thresmapfiledir2],datatype='Float32')
                                
#%%
            #count number of zeros (below threshold)
            pv_mean_under_thres=np.zeros([num_lat,num_lon],dtype=np.float32)
            pv_sum_thres_diff=np.zeros([num_lat,num_lon],dtype=np.float32)
            if thres_under_above=="under_thres":
                for i_lat in range(num_lat):
                    for i_lon in range(num_lon):
                        power_pv_pix=power_pv[:,i_lat,i_lon]
                        power_thres=power_thres_map[i_lat,i_lon]
                        bool_pv_pix_thres=power_pv_pix<power_thres                   
                        power_pv_under_thres_mean=np.mean(power_pv_pix[bool_pv_pix_thres])  
                        power_pv_sum_thres_diff=np.sum(power_thres-power_pv_pix[bool_pv_pix_thres])/num_days_of_year*365
                        pv_mean_under_thres[i_lat,i_lon]=power_pv_under_thres_mean
                        pv_sum_thres_diff[i_lat,i_lon]=power_pv_sum_thres_diff
            else:
                for i_lat in range(num_lat):
                    for i_lon in range(num_lon):
                        power_pv_pix=power_pv[:,i_lat,i_lon]
                        power_thres=power_thres_map[i_lat,i_lon]
                        bool_pv_pix_thres=power_pv_pix>power_thres
                        power_pv_under_thres_mean=np.mean(power_pv_pix[bool_pv_pix_thres])    
                        power_pv_sum_thres_diff=np.sum(power_thres-power_pv_pix[bool_pv_pix_thres])/num_days_of_year*365
                        pv_mean_under_thres[i_lat,i_lon]=power_pv_under_thres_mean
                        pv_sum_thres_diff[i_lat,i_lon]=power_pv_sum_thres_diff
                    
#%%
            #mask out non-land area
            if bool_mask_land:
                land_area=np.array(modelattr.variables["landpart"][:]).astype(np.float32)     
                land_area_mask=np.zeros([num_lat,num_lon],dtype=np.bool_)
                d=skfmm.distance((1-land_area),dx=360/num_lon*111)
                land_area_mask[d<371]=True
                pv_mean_under_thres[land_area_mask==0]=-9999
            
            #the original result is flipped
            ras_thres_mean=np.flipud(pv_mean_under_thres)
            ras_thres_mean_sft=np.zeros([num_lat,num_lon],dtype=np.float32)
            ras_thres_mean_sft[:,0:int(num_lon/2)]=ras_thres_mean[:,int(num_lon/2):num_lon]
            ras_thres_mean_sft[:,int(num_lon/2):num_lon]=ras_thres_mean[:,0:int(num_lon/2)]            
            
            ras_thres_diff=np.flipud(pv_sum_thres_diff)
            ras_thres_diff_sft=np.zeros([num_lat,num_lon],dtype=np.float32)
            ras_thres_diff_sft[:,0:int(num_lon/2)]=ras_thres_diff[:,int(num_lon/2):num_lon]
            ras_thres_diff_sft[:,int(num_lon/2):num_lon]=ras_thres_diff[:,0:int(num_lon/2)]        
            
            #write results
            driver=gdal.GetDriverByName("GTiff")
            proj = osr.SpatialReference()
            proj.ImportFromEPSG(4326)
            proj = proj.ExportToWkt()
            geotrans=[-180,360/num_lon,0,90,0,-1*180/num_lat]
            
            if thres_under_above=="under_thres":
                rasfolderdirto=resultrootfolderdir+os.sep+power_name+"_incl_season"+os.sep+"under_thres_days"+os.sep+period+os.sep+modelName
                rasfilename2=power_name+"_under_thres"+str(int(historical_thres_perc*100)).zfill(3)+"_"+str(year_start)+"_"+str(year_end)+"_pv_mean.tif"     
                rasfilename3=power_name+"_under_thres"+str(int(historical_thres_perc*100)).zfill(3)+"_"+str(year_start)+"_"+str(year_end)+"_ext.tif"                
            else:
                rasfolderdirto=resultrootfolderdir+os.sep+power_name+"_incl_season"+os.sep+"above_thres_days"+os.sep+period+os.sep+modelName
                rasfilename2=power_name+"_above_thres"+str(int(historical_thres_perc*100)).zfill(3)+"_"+str(year_start)+"_"+str(year_end)+"_pv_mean.tif"  
                rasfilename3=power_name+"_above_thres"+str(int(historical_thres_perc*100)).zfill(3)+"_"+str(year_start)+"_"+str(year_end)+"_ext.tif"  
            if not os.path.exists(rasfolderdirto):
                os.makedirs(rasfolderdirto)
            rasfiledir2=rasfolderdirto+os.sep+rasfilename2  
            rasfiledir3=rasfolderdirto+os.sep+rasfilename3 
            # ptf.writeNumpyToTiff(ras_thres_mean_sft,driver,geotrans,proj,num_lat,num_lon,-9999,rasfiledir2,datatype='Float32')
            ptf.writeNumpyToTiff(ras_thres_diff_sft,driver,geotrans,proj,num_lat,num_lon,-9999,rasfiledir3,datatype='Float32')
            print("done.\n")
    print("ALL DONE.\n")
