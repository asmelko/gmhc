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
	clustering::asgn_t i;
	clustering::asgn_t j;
};

struct input_t
{
	float* data;
	size_t count;
	size_t dim;
};

struct kernel_info
{
	unsigned int grid_dim;
	unsigned int block_dim;
	size_t shared_size;
};

void run_euclidean_min(const input_t in, output_t* out, kernel_info info);

#endif