
#main authors: Heng Zhang (heng.zhang@eawag.ch; hengzhang.zhh@gmail.com) & Laibao Liu (laibao@hku.hk)
#Swiss Federal Institute of Aquatic Science and Technology (EAWAG / ETH Domain) & The University of Hong Kong (HKU)
#Please cite the paper when using any part of this code/project. 

library(tidyverse)
library(raster)
library(rnaturalearth)
library(sf)
library(patchwork)

root<-"XXX"
maskfolderdir<-paste0(root,"/","data/GCM_ModelAttr")
mapfolderdir<-paste0(root,"/","result","/","raster")
dirto<-paste0(root,"/","result","/","raster","/","wind_solar_change")
if (dir.exists(dirto)==FALSE){
  dir.create(dirto,recursive = TRUE)
}

csm_models<-c("ACCESS-ESM1-5","BCC-CSM2-MR","CanESM5","CESM2","CMCC-ESM2","EC-Earth3","GFDL-ESM4",
              "IPSL-CM6A-LR","MPI-ESM1-2-HR","MRI-ESM2-0","NorESM2-MM","UKESM1-0-LL")

powers<-c("wind","solar")                           #"solar"    "wind"
metrics<-c("PMean")                             #"PMean"   "PVar"   "PExt"
tar_periods<-c("ssp245","ssp585")                     #ssp245    ssp370    ssp585
tar_year_labels<-c("2066_2095","2036_2065")              #"2066_2195"  "2036_2065"

ref_period<-"historical_1950_2014"
ref_year_label<-"1980_2009"

nut_metric<-"under_thres010" 
num_years<-30

num_csm<-length(csm_models)

nandefault<--9999
robinson <- CRS("+proj=robin +over")
wgs1984<-CRS("+proj=longlat +datum=WGS84")
robinson_reso<-10000
wgs1984_reso<-0.083333333333333

maskfilename<-"land_area_mask.tif"        #"land_area_buffer.tif", "land_area_mask.tif"
maskfiledir<-paste0(maskfolderdir,"/",maskfilename)

ras_mask<-raster(maskfiledir)
ras_mask<-reclassify(ras_mask, cbind(nandefault, NA))
ras_mask<-projectRaster(ras_mask, res=wgs1984_reso, crs = wgs1984)
num_pixels<-length(ras_mask)

bool_land_mask<-(ras_mask>0)
bool_land_mask[bool_land_mask==0]<-NA

for(power in powers){
  for(metric in metrics){
    for(tar_period in tar_periods){
      for(tar_year_label in tar_year_labels){
        
        print(paste0("working on: ",power," ",metric," ",tar_period," ",tar_year_label,"...... "))
        
        resultfilename<-paste0("rate_change_",power,"_",metric,"_",tar_period,"_",tar_year_label,"_",ref_period,".tif")
        resultfiledir<-paste0(dirto,"/",resultfilename)
        if(file.exists(resultfiledir)){
          print("already finished, continue.")
          next
        }
        
        #---------------------- future - present ----------------------
        model_values<-array(NA,dim = c(num_pixels,num_csm))
        if(metric=="PMean"){
          power_prefix<-paste0(power,"_opt_",tolower(metric),"_")
          power_suffix<-".tif"
          for(i_csm in 1:num_csm){
            csm<-csm_models[i_csm]
            #find raster layer directories
            tarfolderdir<-paste0(mapfolderdir,"/","wind_solar_opt_",metric,"/",power,"_no_season/",tar_period,"/",csm)
            reffolderdir<-paste0(mapfolderdir,"/","wind_solar_opt_",metric,"/",power,"_no_season/",ref_period,"/",csm)
            tarfiledir<-paste0(tarfolderdir,"/",power_prefix,tar_year_label,power_suffix)
            reffiledir<-paste0(reffolderdir,"/",power_prefix,ref_year_label,power_suffix)
            #read raster maps
            ras_tar<-raster(tarfiledir)
            ras_ref<-raster(reffiledir)
            
            ras_diff<-ras_tar-ras_ref                   #for PD & PS
            
            ras_diff<-100*ras_diff/(ras_ref+1e-4)                  #calculate the rate of change (%)
            ras_diff<-projectRaster(ras_diff, res=wgs1984_reso, crs = wgs1984)
        
            model_values[,i_csm]<-values(ras_diff)
          }
        }else if(metric=="PExt"){
          power_prefix<-paste0(power,"_",nut_metric,"_")
          power_suffix<-paste0("_ext.tif")
          for(i_csm in 1:num_csm){
            csm<-csm_models[i_csm]
            
            reffolderdir<-paste0(mapfolderdir,"/","wind_solar_TS_analysis","/",
                                 power,"_incl_season/under_thres_days/",ref_period,"/",csm)
            reffiledir<-paste0(reffolderdir,"/",power_prefix,ref_year_label,power_suffix) 
            tarfolderdir<-paste0(mapfolderdir,"/","wind_solar_TS_analysis","/",
                                 power,"_incl_season/under_thres_days/",tar_period,"/",csm)
            tarfiledir<-paste0(tarfolderdir,"/",power_prefix,tar_year_label,power_suffix) 
            
            ras_ref<-raster(reffiledir)/num_years
            ras_tar<-raster(tarfiledir)/num_years
            ras_diff<-100*(ras_tar-ras_ref)/(ras_ref+1e-4)

            ras_diff<-projectRaster(ras_diff, res=wgs1984_reso, crs = wgs1984)
            # ras_diff[is.na(bool_land_mask)]<--9999
            model_values[,i_csm]<-values(ras_diff)
          }          
        }else if(metric=="PVar"){
          #Here to calculate CV
          power_prefix<-paste0(power,"_season_",tolower(metric),"_")
          power_suffix<-".tif"
          ref_values<-array(NA,dim = c(num_pixels,num_csm))
          for(i_csm in 1:num_csm){
            csm<-csm_models[i_csm]
            #find raster layer directories
            #CV for seasonality
            power_prefix<-paste0(power,"_season_",tolower(metric),"_")
            reffolderdir<-paste0(mapfolderdir,"/","wind_solar_season","/",ref_period,"/",csm)
            reffiledir<-paste0(reffolderdir,"/",power_prefix,ref_year_label,power_suffix)
            tarfolderdir<-paste0(mapfolderdir,"/","wind_solar_season","/",tar_period,"/",csm)
            tarfiledir<-paste0(tarfolderdir,"/",power_prefix,tar_year_label,power_suffix)
            #read raster maps
            ras_ref_ssn<-raster(reffiledir)
            ras_tar_ssn<-raster(tarfiledir)
            
            #CV for inter-seasonality
            power_prefix<-paste0(power,"_opt_",tolower(metric),"_")
            reffolderdir<-paste0(mapfolderdir,"/","wind_solar_opt_",metric,"/",power,"_no_season/",ref_period,"/",csm)
            reffiledir<-paste0(reffolderdir,"/",power_prefix,ref_year_label,power_suffix)
            tarfolderdir<-paste0(mapfolderdir,"/","wind_solar_opt_",metric,"/",power,"_no_season/",tar_period,"/",csm)
            tarfiledir<-paste0(tarfolderdir,"/",power_prefix,tar_year_label,power_suffix)
            #read raster maps
            ras_ref_isn<-raster(reffiledir)
            ras_tar_isn<-raster(tarfiledir)
            
            ras_ref<-(ras_ref_ssn+ras_ref_isn)/2
            ras_tar<-(ras_tar_ssn+ras_tar_isn)/2
            
            ras_diff<-ras_tar-ras_ref
            ras_diff<-100*ras_diff/(ras_ref+1e-4)
            ras_diff<-projectRaster(ras_diff, res=wgs1984_reso, crs = wgs1984)
        
            model_values[,i_csm]<-values(ras_diff)
          }
        }
        
        #calculate the median of change
        ras_diff_median<-apply(model_values,1,median)
        ras_csm_median<-ras_diff
        values(ras_csm_median)<-ras_diff_median
        
        ras_num_loss<-apply(model_values<0,1,sum)
        ras_num_gain<-apply(model_values>0,1,sum)
        bool_major_loss<-ras_num_loss>0.66*num_csm
        bool_major_gain<-ras_num_gain>0.66*num_csm
        bool_sign_change<-bool_major_loss|bool_major_gain
        
        ras_sign_change<-ras_diff
        values(ras_sign_change)<-bool_sign_change
        
        resultfilename<-paste0("rate_change_",power,"_",metric,"_",tar_period,"_",tar_year_label,"_",ref_period,".tif")
        resultfiledir<-paste0(dirto,"/",resultfilename)
        writeRaster(ras_csm_median, resultfiledir, format="GTiff", overwrite=TRUE)
        
        resultfilename<-paste0("sign_change_",power,"_",metric,"_",tar_period,"_",tar_year_label,"_",ref_period,".tif")
        resultfiledir<-paste0(dirto,"/",resultfilename)
        writeRaster(ras_sign_change, resultfiledir, format="GTiff", overwrite=TRUE)        

      }
    }
  }
}
