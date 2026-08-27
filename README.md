Program: Climate Change Impacts on Global Wind and Solar Power Plants

Author: Laibao Liu, Mengxi Wu, Florian Altermatt, Gang He, Martin Wild, Yu Liu, Xi Lu, Heng Zhang*
Contact: heng.zhang@eawag.ch; hengzhang.zhh@gmail.com
Date: 08/27/2026

Abstract

The deployment of wind and solar power plants is necessary and effective for mitigating climate change, but their power output depends on wind speed and solar irradiance, respectively, which are themselves affected by weather variability and climate change. To date, however, there is still a lack of a comprehensive global-scale assessment of climate impacts on productivity of committed wind and solar power plants, putting uncertainty on their future effectiveness. Here, we combine the best publicly available global wind and solar power plants database and 12 state-of-the-art global climate models to assess climate change impacts on their daily power output in three dimensions: mean, variance and extreme. We find that ~49% of committed wind capacity (735 GW) and ~8% of committed solar capacity (152 GW) are projected to simultaneously experience decreased power density and increased variance and extremes by 2100 under high-emission scenario, with a high model agreement level (8 out of 12 models). 66% of wind capacity and 47 % of solar capacity are estimated to decrease by 7 % and 3% in the mean of power density, respectively. By contrast, about 7% of committed wind capacity (100 GW) and 31% of committed solar capacity (596 GW) are projected to improve across all three dimensions. 6% of wind capacity and 40% of solar capacity are estimated to increase by 8% and 4% in the mean of power density, respectively. Depending on the geographic location and stability of the electric grid, climate change can thus both dampen or enhance the existing energy system security. Climate-vulnerable wind and solar power plants are mostly distributed in some top carbon emitters, including China, US, and EU27. Meanwhile, Global South might expose more negative climate change impacts on solar energy than the Global North. We identify climate-resilient areas for planning new power plant deployment and call for essential climate adaptation actions for committed wind and solar power plants.


Platform and Packages

This Global Wind Solar Power project is programmed with Python, R, and CUDA (GPU computing). Please make sure that you have already installed NVIDIA CUDA computing environment. We recommend using Anaconda distribution. Please make sure that the following modules have been successfully installed: 

numpy, pandas, gdal, pycuda, skfmm, scikit-learn, scipy

Code and Data for Demo

Python, R and CUDA scripts to compute PMean, PExt, PVar (both inter-seasonal and intra-seasonal CV) are in the "code" folder. These codes can be executed in the following sequence. 

01_calc_wind_solar_PMean_gpu.py: This code calculates the mean of power density for global wind and solar power. NVIDIA GPU is needed. 

02_calc_wind_solar_PExt_cpu.py: This code calculates the extreme (e.g., below threshold) of power density for global wind and solar power. Only CPU is needed. 

03_calc_wind_solar_intra_season_PVar_gpu.py: This code calculates the intra-seasonal coefficient of variation (CV) of global wind and solar power. NVIDIA GPU is needed.

04_calc_wind_solar_seasonal_component_gpu.py: This code calculates the seasonal component of global wind and solar power. The result comprises of a yearly time series of power density for 365 days. NVIDIA GPU is needed.

05_cala_wind_solar_inter_season_PVar_cpu.py: Inter-seasonal CV is calculated based on the seasonal component. Only CPU is needed.

06_calc_wind_solar_change_map.R: The map of change (in ratio) is calculated based on the future and present PMean, PExt, PVar (both intra- and inter-seasonal CV). This program is programmed with R. 

07_extract_raster_power_plant_values.py: This code extracts values of raster layers based on the location of power plants (in the “data/global_wind_solar_plant” folder). Only CPU is needed.

All python functions and CUDA kernel functions are in the "src" folder. 


The time series of simulation date for each of the climate model is in “data/GCM_ModelAttr” folder, which is used in the above PMean, PExt, PVar computing to locate the raster layer along the time dimension. Global land mask for plotting use is also stored in the same folder. Moreover, the basic information and location for all the considered global wind and solar power plants are in the “data/global_wind_solar_plant” folder. 

Reference
Laibao Liu, Mengxi Wu, Florian Altermatt, Gang He, Martin Wild, Yu Liu, Xi Lu, Heng Zhang*. Climate change impacts on productivity of committed onshore wind and solar power plants. 
