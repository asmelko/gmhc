#ifndef COMMON_KERNELS_CUH
#define COMMON_KERNELS_CUH

#include "device_launch_parameters.h"

#include "kernels.cuh"

__inline__ __device__ float euclidean_norm(const float* __restrict__ sub, clustering::csize_t dim)
{
	float tmp = 0;
	for (size_t i = 0; i < dim; ++i)
		tmp += sub[i] * sub[i];

	return sqrtf(tmp);
}

__device__ float euclidean_norm(const float* __restrict__ l_point, const float* __restrict__ r_point, clustering::csize_t dim);

__device__ void reduce_sum_warp(float* __restrict__ point, clustering::csize_t dim);
__device__ void reduce_sum_block(float* __restrict__ point, clustering::csize_t dim, float* __restrict__ shared_mem);

__device__ chunk_t reduce_min_warp(chunk_t data);
__device__ chunk_t reduce_min_block(chunk_t data, chunk_t* shared_mem);


__device__ csize2 compute_coordinates(clustering::csize_t count_in_line, clustering::csize_t plain_index);

#endif