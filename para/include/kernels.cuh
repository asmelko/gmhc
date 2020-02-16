#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <clustering.hpp>

#include <cstdint>
#include <cuda_runtime.h>

#define CUCH(x) cuda_check(x, __FILE__, __LINE__)
void cuda_check(cudaError_t code, const char* file, int line);

struct output_t
{
	float distance;
	clustering::asgn_t pair[2];
};

struct input_t
{
	float* data;
	size_t dim;
	size_t count;
};

__global__ void euclidean_min(const float* points, size_t point_count, size_t point_dim, size_t shared_size, output_t* res);

__global__ void reduce_min(output_t* res, size_t output_size);

#endif