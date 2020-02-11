#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
#include <clustering.hpp>

#define CUCH(x) cuda_check(x, __FILE__, __LINE__)

struct output_t
{
	float distance;
	std::uint64_t pair;
};

struct input_t
{
	float* data;
	size_t dim;
	size_t count;
};

void cuda_check(cudaError_t code, const char* file, int line);

void run_euclid_max();

#endif