#include <device_launch_parameters.h>

#include "kernels.cuh"
#include "neighbor_common.cuh"

using namespace clustering;

template<csize_t N>
__inline__ __device__ void point_neighbors_thread(neighbor_t* __restrict__ neighbors,
    const float* __restrict__ this_centroid,
    const float* __restrict__ curr_centroid,
    csize_t dim,
    csize_t idx)
{
    float dist = euclidean_norm(this_centroid, curr_centroid, dim);

    if ((isinf(dist) || isnan(dist)))
        dist = FLT_MAX;

    add_neighbor<N>(neighbors, neighbor_t { dist, idx });
}

template<csize_t N>
__device__ void point_neighbor(const float* __restrict__ centroids,
    neighbor_t* __restrict__ neighbors,
    neighbor_t* __restrict__ work_neighbors,
    float* __restrict__ shared_mem,
    csize_t dim,
    csize_t count,
    csize_t x,
    bool is_new)
{
    neighbor_t local_neighbors[N];

    for (csize_t i = 0; i < N; ++i)
        local_neighbors[i].distance = FLT_INF;

    for (csize_t i = threadIdx.x; i < dim; i += blockDim.x)
        shared_mem[i] = centroids[x * dim + i];

    __syncthreads();

    csize_t y = threadIdx.x + blockIdx.x * blockDim.x;

    if (is_new)
    {
        for (; y < x; y += blockDim.x * gridDim.x)
            point_neighbors_thread<N>(neighbors + y * N, shared_mem, centroids + y * dim, dim, x);

        y += 1;
    }
    else
        y += x + 1;

    for (; y < count; y += blockDim.x * gridDim.x)
        point_neighbors_thread<N>(local_neighbors, shared_mem, centroids + y * dim, dim, y);

    reduce_min_block<N>(local_neighbors, reinterpret_cast<neighbor_t*>(shared_mem + dim));

    if (threadIdx.x == 0)
        memcpy(work_neighbors + (gridDim.x * x + blockIdx.x) * N, local_neighbors, N * sizeof(neighbor_t));
}

template<csize_t N>
__global__ void neighbors(
    const float* __restrict__ centroids, neighbor_t* __restrict__ neighbors, csize_t dim, csize_t count)
{
    extern __shared__ float shared_mem[];

    for (csize_t x = 0; x < count; ++x)
        point_neighbor<N>(centroids, nullptr, neighbors, shared_mem, dim, count, x, false);
}

template<csize_t N>
__global__ void neighbors_u(const float* __restrict__ centroids,
    neighbor_t* __restrict__ neighbors,
    neighbor_t* __restrict__ work_neighbors,
    const csize_t* __restrict__ updated,
    const csize_t* __restrict__ upd_count,
    csize_t dim,
    csize_t count)
{
    extern __shared__ float shared_mem[];

    auto update_count = *upd_count;

    for (csize_t i = 0; i < update_count; ++i)
        point_neighbor<N>(centroids, neighbors, work_neighbors, shared_mem, dim, count, updated[i], false);
}

template<csize_t N>
__global__ void neighbors_u(const float* __restrict__ centroids,
    neighbor_t* __restrict__ neighbors,
    neighbor_t* __restrict__ work_neighbors,
    csize_t new_idx,
    csize_t dim,
    csize_t count)
{
    extern __shared__ float shared_mem[];

    point_neighbor<N>(centroids, neighbors, work_neighbors, shared_mem, dim, count, new_idx, true);
}
