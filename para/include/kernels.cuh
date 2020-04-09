#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <clustering.hpp>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "common_structures.hpp"

#define CUCH(x) cuda_check(x, __FILE__, __LINE__)
void cuda_check(cudaError_t code, const char* file, int line);

#define BUCH(x) cuBLAS_check(x, __FILE__, __LINE__)
void cuBLAS_check(cublasStatus_t code, const char* file, int line);

#ifdef __INTELLISENSE__
void __syncthreads() {}
void __syncwarp() {}
unsigned __ballot_sync(unsigned mask, bool predicate);
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

constexpr size_t MAX_DIM = 50;

void assign_constant_storage(const float* value, size_t size, cudaMemcpyKind kind);

chunk_t run_euclidean_min(const input_t in, chunk_t* out, const float* const* inverses, kernel_info info);
void run_min(const input_t in, chunk_t* out, const float* const* inverses, kernel_info info);
chunk_t run_reduce(const chunk_t* chunks, chunk_t* out, size_t chunk_size, kernel_info info);

void run_centroid(const input_t in, const clustering::asgn_t* assignments, float* out, clustering::asgn_t cetroid_id, size_t cluster_size, kernel_info info);

void run_covariance(const input_t in, const clustering::asgn_t* assignments, float* out, clustering::asgn_t centroid_id, kernel_info info);
void run_finish_covariance(const float* in_cov_matrix, size_t divisor, size_t N, float* out_cov_matrix);

void run_set_default_inverse(float* icov_matrix, size_t size);
void run_set_default_asgn(clustering::asgn_t* asgns, size_t N);

void run_merge_clusters(clustering::asgn_t* assignments, size_t point_size, clustering::asgn_t old_A, clustering::asgn_t old_B, clustering::asgn_t new_C, kernel_info info);

template <size_t N>
void run_neighbours(const float* centroids, size_t dim, size_t centroid_count, neighbour_t* tmp_neighbours, neighbour_t* neighbours,
	cluster_kind* cluster_kinds, kernel_info info);

template <size_t N>
chunk_t run_neighbours_min(const neighbour_t* neighbours, size_t count, chunk_t* result);

template <size_t N>
void run_update_neighbours(const float* centroids, const float* const* inverses, size_t dim, size_t centroid_count, neighbour_t* tmp_neighbours, neighbour_t* act_neighbours, cluster_kind* cluster_kinds, uint8_t* updated, size_t old_i, size_t old_j, kernel_info info);

void print_nei(neighbour_t* neighbours, size_t nei_number, size_t count);
void run_print_assg(clustering::asgn_t* assignments, size_t point_size);
void run_print_centroid(const float* centroid, size_t dim, size_t count);
void run_print_up(uint8_t* updated, size_t count);
void run_print_kind(cluster_kind* kind, size_t count);

chunk_t run_simple_min(const float* clusters, size_t dim, size_t count, chunk_t* out);


#endif