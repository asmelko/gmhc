#include <device_launch_parameters.h>

#include "kernels.cuh"
#include "neighbor_common.cuh"

using namespace clustering;

__inline__ __device__ float maha_dist(
    const float* __restrict__ point, const float* __restrict__ matrix, csize_t size, unsigned int lane_id)
{
    float tmp_point = 0;

    auto icov_size = (size + 1) * size / 2;

    for (csize_t i = lane_id; i < icov_size; i += warpSize)
    {
        auto coords = compute_coordinates(size, i);

        tmp_point = fmaf(matrix[i], point[coords.x] * point[coords.y], tmp_point);
    }

    reduce_sum_warp(&tmp_point, 1);

    if (lane_id == 0)
    {
        if (tmp_point < 0)
            return euclidean_norm(point, size);
        return sqrtf(tmp_point);
    }
    return 0;
}

template<csize_t N>
__inline__ __device__ void point_neighbors_mat_warp(const float* __restrict__ centroids,
    const float* __restrict__ inverses,
    neighbor_t* __restrict__ neighbors,
    float* __restrict__ curr_centroid,
    const float* __restrict__ this_centroid,
    const float* __restrict__ this_icov,
    csize_t dim,
    csize_t idx)
{
    float dist = 0;

    auto warp_id = threadIdx.x / warpSize;
    auto lane_id = threadIdx.x % warpSize;

    for (csize_t i = lane_id; i < dim; i += warpSize)
        curr_centroid[warp_id * dim + i] = centroids[idx * dim + i] - this_centroid[i];

    __syncwarp();

    auto icov_size = (dim + 1) * dim / 2;
    dist += maha_dist(curr_centroid + warp_id * dim, inverses + idx * icov_size, dim, lane_id);

    dist += maha_dist(curr_centroid + warp_id * dim, this_icov, dim, lane_id);

    if (lane_id == 0)
    {
        dist = (isinf(dist) || isnan(dist)) ? FLT_MAX : dist / 2;

        add_neighbor<N>(neighbors, neighbor_t { dist, idx });
    }
}

template<csize_t N>
__inline__ __device__ void point_neighbors_mat(const float* __restrict__ centroids,
    const float* __restrict__ inverses,
    neighbor_t* __restrict__ neighbors,
    float* __restrict__ shared_mem,
    csize_t dim,
    csize_t count,
    csize_t x)
{
    float* this_centroid = shared_mem;
    float* this_icov = shared_mem + dim;
    float* curr_centroid = shared_mem + dim + dim * dim;

    neighbor_t local_neighbors[N];

    for (csize_t i = 0; i < N; ++i)
        local_neighbors[i].distance = FLT_INF;

    auto icov_size = (dim + 1) * dim / 2;

    for (csize_t i = threadIdx.x; i < dim + icov_size; i += blockDim.x)
    {
        if (i < dim)
            shared_mem[i] = centroids[x * dim + i];
        else
            shared_mem[i] = inverses[x * icov_size + i - dim];
    }

    __syncthreads();

    for (csize_t idx = threadIdx.x + blockIdx.x * blockDim.x + (x + 1) * warpSize; idx < count * warpSize;
         idx += blockDim.x * gridDim.x)
    {
        point_neighbors_mat_warp<N>(
            centroids, inverses, local_neighbors, curr_centroid, this_centroid, this_icov, dim, idx / warpSize);
    }

    __syncthreads();

    reduce_min_block<N>(local_neighbors, reinterpret_cast<neighbor_t*>(shared_mem), false);

    __syncthreads();

    if (threadIdx.x == 0)
        memcpy(neighbors + (gridDim.x * x + blockIdx.x) * N, local_neighbors, sizeof(neighbor_t) * N);
}

template<csize_t N>
__global__ void neighbors_mat(const float* __restrict__ centroids,
    const float* __restrict__ inverses,
    neighbor_t* __restrict__ neighbors,
    csize_t dim,
    csize_t count)
{
    extern __shared__ float shared_mem[];

    for (csize_t x = 0; x < count; ++x)
        point_neighbors_mat<N>(centroids, inverses, neighbors, shared_mem, dim, count, x);
}

template<csize_t N>
__global__ void neighbors_mat_u(const float* __restrict__ centroids,
    const float* __restrict__ inverses,
    neighbor_t* __restrict__ neighbors,
    csize_t* __restrict__ updated,
    const csize_t* __restrict__ upd_count,
    csize_t dim,
    csize_t count)
{
    extern __shared__ float shared_mem[];

    auto update_count = *upd_count;

    for (csize_t i = 0; i < update_count; ++i)
        point_neighbors_mat<N>(centroids, inverses, neighbors, shared_mem, dim, count, updated[i]);
}
