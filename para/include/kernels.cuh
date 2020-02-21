#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <clustering.hpp>
#include <cuda_runtime.h>

#define CUCH(x) cuda_check(x, __FILE__, __LINE__)
void cuda_check(cudaError_t code, const char* file, int line);

constexpr size_t MAX_DIM = 50;

#ifdef __INTELLISENSE__
void __syncthreads() {}
unsigned __activemask() {}
template <typename T>
T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width = warpSize) {}
template <typename T>
T atomicAdd(T* address, T val) {}
#endif

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

void assign_constant_storage(const float* value, size_t size, cudaMemcpyKind kind);

void run_euclidean_min(const input_t in, output_t* out, kernel_info info);

void run_centroid(const input_t in, const clustering::asgn_t* assignments, float* out, clustering::asgn_t cetroid_id, kernel_info info);

void run_covariance(const input_t in, const clustering::asgn_t* assignments, float* out, clustering::asgn_t centroid_id, kernel_info info);


#endif