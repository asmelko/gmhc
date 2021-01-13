#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <limits>

#include "clustering.hpp"
#include "structures.hpp"

#define CUCH(x) cuda_check(x, __FILE__, __LINE__)
#define SOCH(x) cuSOLVER_check(x, __FILE__, __LINE__)

// check that halts the program if there was a cuda error
void cuda_check(cudaError_t code, const char* file, int line);
// check that halts the program if there was a cusolver error
void cuSOLVER_check(cusolverStatus_t code, const char* file, int line);

#ifdef __INTELLISENSE__
void __syncthreads() {}
void __syncwarp() {}
unsigned __ballot_sync(unsigned mask, bool predicate);
unsigned __activemask() {}
template<typename T>
T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width = warpSize)
{}
template<typename T>
T atomicAdd(T* address, T val)
{}
#endif

// constant that states the maximal allowed dimension of a point
constexpr clustering::csize_t MAX_DIM = 50;
// constant that represents infinity
constexpr float FLT_INF = std::numeric_limits<float>::infinity();

chunk_t run_euclidean_min(const input_t in, chunk_t* out, const float* const* inverses, kernel_info info);
void run_min(const input_t in, chunk_t* out, const float* const* inverses, kernel_info info);
chunk_t run_reduce(const chunk_t* chunks, chunk_t* out, clustering::csize_t chunk_size, kernel_info info);

// computes centroid of a cluster
void run_centroid(const float* points,
    const clustering::asgn_t* assignments,
    float* work_centroid,
    float* out_centroid,
    clustering::csize_t dim,
    clustering::csize_t point_count,
    clustering::asgn_t cluster_id,
    clustering::csize_t cluster_size,
    kernel_info info);

// assigns data from an array of specific size to constant storage
void assign_constant_storage(
    const float* value, clustering::csize_t size, cudaMemcpyKind kind, cudaStream_t stream = (cudaStream_t)0);

// computes covariance of a cluster
void run_covariance(const float* points,
    const clustering::asgn_t* assignments,
    float* work_covariance,
    float* out_covariance,
    clustering::csize_t dim,
    clustering::csize_t point_count,
    clustering::asgn_t centroid_id,
    clustering::csize_t divisor,
    kernel_info info);
// runs store_covariance kernel
void run_store_icovariance_data(float* icov_dest,
    float* mf_dest,
    const float* icov_src,
    const float mf_src,
    clustering::csize_t dim,
    cudaStream_t stream = (cudaStream_t)0);

void run_transform_cov(float* matrix,
    clustering::csize_t dim,
    float weight_factor,
    bool use_cholesky,
    const float* cholesky_decomp,
    const int* cholesky_success,
    cudaStream_t stream = (cudaStream_t)0);

void run_compute_store_icov_mf(
    float* dest, clustering::csize_t dim, const float* cholesky_decomp, cudaStream_t stream = (cudaStream_t)0);


// updates assignment array to merge clusters
void run_merge_clusters(clustering::asgn_t* assignments,
    clustering::csize_t point_size,
    clustering::asgn_t old_A,
    clustering::asgn_t old_B,
    clustering::asgn_t new_C,
    kernel_info info);

// initializes the neighbor array
template<clustering::csize_t N>
void run_neighbors(centroid_data_t data,
    neighbor_t* tmp_neighbors,
    neighbor_t* act_neighbors,
    clustering::csize_t size,
    bool use_eucl,
    kernel_info info);

// retrieves the minimum from a neighbor array
template<clustering::csize_t N>
chunk_t run_neighbors_min(const neighbor_t* neighbors, clustering::csize_t size, chunk_t* result);

// updates neighbor array
template<clustering::csize_t N>
void run_update_neighbors(centroid_data_t data,
    neighbor_t* tmp_neighbors,
    neighbor_t* act_neighbors,
    clustering::csize_t size,
    update_data_t upd_data,
    bool use_eucl,
    kernel_info info);

// updates neighbor array of a new cluster
template<clustering::csize_t N>
void run_update_neighbors_new(centroid_data_t data,
    neighbor_t* tmp_neighbors,
    neighbor_t* act_neighbors,
    clustering::csize_t size,
    clustering::csize_t new_idx,
    bool use_eucl,
    kernel_info info);

// sets identity matrix
void run_set_unit_matrix(float* matrix, clustering::csize_t size, cudaStream_t stream = (cudaStream_t)0);
// sets initial assignments
void run_set_default_asgn(clustering::asgn_t* asgns, clustering::csize_t N);
// sets initial inverse covariances
void run_set_default_icovs(
    float* __restrict__ icovs, clustering::csize_t size, clustering::csize_t point_dim, kernel_info info);
// sets initial multiplication factors of icovs
void run_set_default_icov_mfs(float* __restrict__ mfs, clustering::csize_t size, kernel_info info);



// debug kernel - prints neighbor array
void run_print_nei(neighbor_t* neighbors, clustering::csize_t nei_number, clustering::csize_t count);
// debug kernel - prints assignment array
void run_print_assg(clustering::asgn_t* assignments, clustering::csize_t point_size);
// debug kernel - prints centroid array
void run_print_centroid(const float* centroid, clustering::csize_t dim, clustering::csize_t count);
// debug kernel - prints update array
void run_print_up(clustering::csize_t* updated, clustering::csize_t* count);
// debug kernel - computes euclidean distance
float run_point_eucl(const float* lhs_centroid, const float* rhs_centroid, clustering::csize_t dim);
// debug kernel - computes maha distance
float run_point_maha(const float* lhs_centroid,
    const float* rhs_centroid,
    clustering::csize_t dim,
    const float* lhs_icov,
    const float* rhs_icov,
    const float* lhs_mf,
    const float* rhs_mf);
// debug kernel - compares two neighbor arrays - only updated part
void run_compare_nei_u(const neighbor_t* lhs,
    const neighbor_t* rhs,
    const clustering::csize_t* update,
    const clustering::csize_t* size,
    clustering::csize_t new_idx);
// debug kernel - compares two neighbor arrays
void run_compare_nei(const neighbor_t* lhs,
    const neighbor_t* rhs,
    const clustering::csize_t small_size,
    clustering::csize_t big_begin,
    clustering::csize_t big_size,
    clustering::csize_t new_idx);
// debug kernel - trivial minimum retrieval algorithm
chunk_t run_simple_min(const float* clusters, clustering::csize_t dim, clustering::csize_t count, chunk_t* out);

#endif