
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <cstdio>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define INDEX2D_yx(ncol, idy, idx) ((idy) * (ncol) + (idx))
#define INDEX3D_zyx(nrow, ncol, idz, idy, idx) ((idz) * (nrow) * (ncol) + (idy) * (ncol) + (idx))

#define MAX_TS_LEN 1024

__device__ float calcTSMean(const float* ndary, int2 pixel, const int i_start, const int i_end, const int num_lat, const int num_lon) {
	float ts_mean = 0.0;
#pragma unroll
	for (int i_ts = i_start; i_ts < i_end; i_ts++) {
		ts_mean += ndary[INDEX3D_zyx(num_lat, num_lon, i_ts, pixel.y, pixel.x)];
	}
	ts_mean = ts_mean / (float)(i_end - i_start);
	return ts_mean;
}

__global__ void calcMovWinAvgTrend(float *ndary_trend, float *ndary_res, float *arr_trend_mean, const float* ndary, const int mv_size, const int num_ori_time, const int num_lat, const int num_lon)
{
	int2 pixel;
	pixel.x = threadIdx.x + blockDim.x * blockIdx.x;
	pixel.y = threadIdx.y + blockDim.y * blockIdx.y;
	if (pixel.x >= num_lon || pixel.y >= num_lat)
		return;
	else {
		float arr_mean = 0.0;
#pragma unroll
		for (int i_ts = 0; i_ts < num_ori_time - mv_size; i_ts++) {
			arr_mean = calcTSMean(ndary, pixel, i_ts, i_ts + mv_size, num_lat, num_lon);
			ndary_trend[INDEX3D_zyx(num_lat, num_lon, i_ts, pixel.y, pixel.x)] = arr_mean;
			ndary_res[INDEX3D_zyx(num_lat, num_lon, i_ts, pixel.y, pixel.x)] = ndary[INDEX3D_zyx(num_lat, num_lon, i_ts + mv_size / 2, pixel.y, pixel.x)] - arr_mean;
		}		
		arr_trend_mean[INDEX2D_yx(num_lon, pixel.y, pixel.x)] = calcTSMean(ndary_trend, pixel, 0, num_ori_time - mv_size, num_lat, num_lon);
	}
}

__global__ void calcSeasonalEffect(float* ndary_season, float* ndary_res, const float* ndary, const int num_days_of_year, const int num_time, const int num_lat, const int num_lon) {
	int2 pixel;
	pixel.x = threadIdx.x + blockDim.x * blockIdx.x;
	pixel.y = threadIdx.y + blockDim.y * blockIdx.y;
	if (pixel.x >= num_lon || pixel.y >= num_lat)
		return;
	else {
		int ts_index, day_count;
		float ts_mean = 0.0;
#pragma unroll
		for (int i_day = 0; i_day < num_days_of_year; i_day++) {
			ts_mean = 0.0;
			day_count = 0;
#pragma unroll
			for (int i_ts = i_day; i_ts < num_time; i_ts = i_ts + num_days_of_year) {
				ts_mean += ndary[INDEX3D_zyx(num_lat, num_lon, i_ts, pixel.y, pixel.x)];
				day_count++;
			}
			ts_mean = ts_mean / day_count;
			ndary_season[INDEX3D_zyx(num_lat, num_lon, i_day, pixel.y, pixel.x)] = ts_mean;
#pragma unroll
			for (int i_ts = i_day; i_ts < num_time; i_ts = i_ts + num_days_of_year) {
				ts_index = INDEX3D_zyx(num_lat, num_lon, i_ts, pixel.y, pixel.x);
				ndary_res[ts_index] = ndary[ts_index] - ts_mean;
			}
		}
	}
}

__global__ void calcWSOPTRatioCV(float* ws_opt, float* cv_opt, float* wind_pv, float* solar_pv, const float* wind_pv_mean, const float* solar_pv_mean, const float* ws_ratios, int num_ws_ratios, int num_time, int num_lat, int num_lon)
{
	int2 pixel;
	pixel.x = threadIdx.x + blockDim.x * blockIdx.x;
	pixel.y = threadIdx.y + blockDim.y * blockIdx.y;
	if (pixel.x >= num_lon || pixel.y >= num_lat)
		return;
	else {
		float pv_CVs[MAX_TS_LEN] = { 0.0 };
        float pv_CV, pv_CV_min = 999999;
		float ws_ratio, ts_mean, ts_sum, ts_value;
		int ts_index, cv_index = -1;
		for (int i_wsr = 0; i_wsr < num_ws_ratios; i_wsr++) {
			ws_ratio = ws_ratios[i_wsr];
			//step 1: to calculate the mean of wind + solar (ws_ratio)
			ts_mean = 0.0;
#pragma unroll
			for (int i_ts = 0; i_ts < num_time; i_ts++) {
				ts_index = INDEX3D_zyx(num_lat, num_lon, i_ts, pixel.y, pixel.x);
				ts_mean += ws_ratio * wind_pv[ts_index] + (1 - ws_ratio) * solar_pv[ts_index];
			}
			ts_mean = ts_mean / (float)num_time;
			//step 2: to calculate the standard deviation of wind + solar (ws_ratio)
			ts_sum = 0.0;
#pragma unroll
			for (int i_ts = 0; i_ts < num_time; i_ts++) {
				ts_index = INDEX3D_zyx(num_lat, num_lon, i_ts, pixel.y, pixel.x);
				ts_value = ws_ratio * wind_pv[ts_index] + (1 - ws_ratio) * solar_pv[ts_index];
				ts_sum += (ts_value - ts_mean) * (ts_value - ts_mean);
			}
			//step 3: to calculate the STD and coefficient of variation (CV)
			ts_mean = ws_ratio * wind_pv_mean[INDEX2D_yx(num_lon, pixel.y, pixel.x)] 
            			+ (1 - ws_ratio) * solar_pv_mean[INDEX2D_yx(num_lon, pixel.y, pixel.x)];
			pv_CV = sqrtf(ts_sum / (float)num_time) / (ts_mean + 1e-6);
			pv_CVs[i_wsr] = pv_CV; 
			if(pv_CV < pv_CV_min){
        			pv_CV_min = pv_CV;
        			cv_index = i_wsr;
			}
		}
		//cv_index = findMinIndex(pv_CVs, num_ws_ratios);
		cv_opt[INDEX2D_yx(num_lon, pixel.y, pixel.x)] = pv_CVs[cv_index];
		ws_opt[INDEX2D_yx(num_lon, pixel.y, pixel.x)] = ws_ratios[cv_index];
	}
}

