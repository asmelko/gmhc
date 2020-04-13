#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <clustering.hpp>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "structures.hpp"

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

struct csize2
{
	clustering::csize_t x, y;
};

constexpr clustering::csize_t MAX_DIM = 50;

void assign_constant_storage(const float* value, clustering::csize_t size, cudaMemcpyKind kind);

chunk_t run_euclidean_min(const input_t in, chunk_t* out, const float* const* inverses, kernel_info info);
void run_min(const input_t in, chunk_t* out, const float* const* inverses, kernel_info info);
chunk_t run_reduce(const chunk_t* chunks, chunk_t* out, clustering::csize_t chunk_size, kernel_info info);

void run_merge_clusters(clustering::asgn_t* assignments, clustering::csize_t* assignment_idxs, clustering::csize_t* idxs_size,
	clustering::csize_t point_size, clustering::asgn_t old_A, clustering::asgn_t old_B, clustering::asgn_t new_C, kernel_info info);

void run_centroid(const input_t in, const clustering::asgn_t* assignment_idxs, clustering::csize_t cluster_size, float* out, kernel_info info);

void run_covariance(const input_t in, float* out, const clustering::csize_t* assignment_idxs, clustering::csize_t idx_count, kernel_info info);
void run_finish_covariance(const float* in_cov_matrix, clustering::csize_t divisor, clustering::csize_t dim, float* out_cov_matrix);
void run_store_icovariance(float* dest, const float* src, clustering::csize_t dim);

void run_set_default_inverse(float* icov_matrix, clustering::csize_t size);
void run_set_default_asgn(clustering::asgn_t* asgns, clustering::csize_t N);

template <clustering::csize_t N>
void run_neighbours(const float* centroids, clustering::csize_t dim, clustering::csize_t centroid_count, neighbour_t* tmp_neighbours, neighbour_t* neighbours, kernel_info info);

template <clustering::csize_t N>
chunk_t run_neighbours_min(const neighbour_t* neighbours, cluster_bound_t sizes, chunk_t* result);

template <clustering::csize_t N>
void run_update_neighbours(centroid_data_t data, neighbour_t* tmp_neighbours, neighbour_t* act_neighbours, cluster_bound_t sizes, update_data_t upd_data, kernel_info info);


void print_nei(neighbour_t* neighbours, clustering::csize_t nei_number, clustering::csize_t count);
void run_print_assg(clustering::asgn_t* assignments, clustering::csize_t point_size);
void run_print_centroid(const float* centroid, clustering::csize_t dim, clustering::csize_t count);
void run_print_up(clustering::csize_t* updated, clustering::csize_t* eucl_count, clustering::csize_t maha_begin, clustering::csize_t* maha_count);


chunk_t run_simple_min(const float* clusters, clustering::csize_t dim, clustering::csize_t count, chunk_t* out);


#endif