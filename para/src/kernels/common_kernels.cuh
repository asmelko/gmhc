#ifndef COMMON_KERNELS_CUH
#define COMMON_KERNELS_CUH

#include <device_launch_parameters.h>

#include <kernels.cuh>

__device__ float euclidean_norm(const float* l_point, const float* r_point, size_t dim);

__device__ void reduce_sum_warp(float* point, size_t dim);
__device__ void reduce_sum_block(float* point, size_t dim, float* shared_mem);

__device__ chunk_t reduce_min_warp(chunk_t data);
__device__ chunk_t reduce_min_block(chunk_t data, chunk_t* shared_mem);


__device__ size2 compute_coordinates(size_t count_in_line, size_t plain_index);

#endif