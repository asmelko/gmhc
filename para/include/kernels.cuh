#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <gmhc.hpp>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUCH(x) cuda_check(x, __FILE__, __LINE__)
void cuda_check(cudaError_t code, const char* file, int line);

#define BUCH(x) cuBLAS_check(x, __FILE__, __LINE__)
void cuBLAS_check(cublasStatus_t code, const char* file, int line);

constexpr size_t MAX_DIM = 50;

#ifdef __INTELLISENSE__
void __syncthreads() {}
unsigned __activemask() {}
template <typename T>
T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width = warpSize) {}
template <typename T>
T atomicAdd(T* address, T val) {}
#endif

struct size2
{
	size_t x, y;
};

struct neighbour_t
{
	float distance;
	clustering::asgn_t idx;
};

template <size_t N>
struct neighbour_array_t
{
	neighbour_t neighbours[N];
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

void run_euclidean_min(const input_t in, clustering::chunk_t* out, const float* const* inverses, kernel_info info);
void run_min(const input_t in, clustering::chunk_t* out, const float* const* inverses, kernel_info info);
clustering::chunk_t run_reduce(const clustering::chunk_t* chunks, clustering::chunk_t* out, size_t chunk_size, kernel_info info);

void run_centroid(const input_t in, const clustering::asgn_t* assignments, float* out, clustering::asgn_t cetroid_id, size_t cluster_size, kernel_info info);

void run_covariance(const input_t in, const clustering::asgn_t* assignments, float* out, clustering::asgn_t centroid_id, kernel_info info);
void run_finish_covariance(const float* in_cov_matrix, size_t divisor, size_t N, float* out_cov_matrix);

void run_set_default_asgn(clustering::asgn_t* asgns, size_t N);
void run_set_default_asgn(clustering::centroid_data_t* asgns, size_t N);

void run_merge_clusters(clustering::asgn_t* assignments, size_t point_size, clustering::asgn_t old_A, clustering::asgn_t old_B, clustering::asgn_t new_C, kernel_info info);

#endif