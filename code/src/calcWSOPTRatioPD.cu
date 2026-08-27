
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

extern "C" {
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define INDEX2D_yx(ncol, idy, idx) ((idy) * (ncol) + (idx))
#define INDEX3D_zyx(nrow, ncol, idz, idy, idx) ((idz) * (nrow) * (ncol) + (idy) * (ncol) + (idx))


__global__ void calcWSOPTRatioPD(float* ws_opt, float* pd_opt, float* wind_pv, float* solar_pv, float* ws_ratios, int num_ws_ratios, int num_time, int num_lat, int num_lon)
{
	int2 pixel;
	pixel.x = threadIdx.x + blockDim.x * blockIdx.x;
	pixel.y = threadIdx.y + blockDim.y * blockIdx.y;
	if (pixel.x >= num_lon || pixel.y >= num_lat)
		return;
	else {
		float pv_PD, pv_PD_max = -999999;
		float ws_ratio;
		int ts_index, pd_index = -1;
#pragma unroll
		for (int i_wsr = 0; i_wsr < num_ws_ratios; i_wsr++) {
			ws_ratio = ws_ratios[i_wsr];
			pv_PD = 0.0;
#pragma unroll
			for (int i_ts = 0; i_ts < num_time; i_ts++) {
				ts_index = INDEX3D_zyx(num_lat, num_lon, i_ts, pixel.y, pixel.x);
				pv_PD += ws_ratio * wind_pv[ts_index] + (1 - ws_ratio) * solar_pv[ts_index];
			}
			pv_PD = pv_PD / num_time;
			if (pv_PD > pv_PD_max) {
				pv_PD_max = pv_PD;
				pd_index = i_wsr;
			}
		}
		//cv_index = findMinIndex(pv_CVs, num_ws_ratios);
		pd_opt[INDEX2D_yx(num_lon, pixel.y, pixel.x)] = pv_PD_max;
		ws_opt[INDEX2D_yx(num_lon, pixel.y, pixel.x)] = ws_ratios[pd_index];
	}
}
}
