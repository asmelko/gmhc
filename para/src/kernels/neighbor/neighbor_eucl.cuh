#include <device_launch_parameters.h>

#include "kernels.cuh"
#include "neighbor_common.cuh"
#include "reduce.cuh"

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
__global__ void point_neighbor(const float* __restrict__ centroids,
    neighbor_t* __restrict__ neighbors,
    neighbor_t* __restrict__ work_neighbors,
    csize_t dim,
    csize_t count,
    csize_t x,
    csize_t new_idx)
{
    extern __shared__ float shared_mem[];

    neighbor_t local_neighbors[N];

    for (csize_t i = 0; i < N; ++i)
        local_neighbors[i].distance = FLT_INF;

    for (csize_t i = threadIdx.x; i < dim; i += blockDim.x)
        shared_mem[i] = centroids[x * dim + i];

    __syncthreads();

    csize_t y = threadIdx.x + blockIdx.x * blockDim.x;

    if (x == new_idx)
    {
        for (; y < x; y += blockDim.x * gridDim.x)
            point_neighbors_thread<N>(neighbors + y * N, shared_mem, centroids + y * dim, dim, x);

        y += 1;
    }
    else
        y += x + 1;

    for (; y < count; y += blockDim.x * gridDim.x)
    {
        if (y == new_idx)
            continue;
        point_neighbors_thread<N>(local_neighbors, shared_mem, centroids + y * dim, dim, y);
    }

    reduce_min_block<N>(local_neighbors, reinterpret_cast<neighbor_t*>(shared_mem + dim));

    if (threadIdx.x == 0)
        memcpy(work_neighbors + (gridDim.x * x + blockIdx.x) * N, local_neighbors, N * sizeof(neighbor_t));
}

template<csize_t N>
__global__ void neighbors(const float* __restrict__ centroids,
    neighbor_t* __restrict__ neighbors,
    neighbor_t* __restrict__ work_neighbors,
    csize_t dim,
    csize_t count,
    csize_t thread_limit,
    csize_t block_limit,
    csize_t shared_size)
{
    for (csize_t x = 0; x < count; ++x)
    {
        auto work = count - x;

        constexpr size_t factor = 4;
        auto tmp = (work + factor - 1) / factor;
        auto threads = ((tmp + warpSize - 1) / warpSize) * warpSize;
        auto blocks = (threads + thread_limit - 1) / thread_limit;

        threads = threads > thread_limit ? thread_limit : threads;
        blocks = blocks > block_limit ? block_limit : blocks;

        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);

        point_neighbor<N><<<blocks, threads, shared_size, s>>>(centroids, nullptr, work_neighbors, dim, count, x, 0);
        reduce_u<N><<<1, 32, 0, s>>>(work_neighbors, neighbors, x, blocks);

        cudaStreamDestroy(s);
    }
}

template<csize_t N>
__global__ void neighbors_u(const float* __restrict__ centroids,
    neighbor_t* __restrict__ neighbors,
    neighbor_t* __restrict__ work_neighbors,
    const csize_t* __restrict__ updated,
    const csize_t* __restrict__ upd_count,
    csize_t new_idx,
    csize_t dim,
    csize_t count,
    csize_t max_threads,
    csize_t max_blocks,
    csize_t shared_size)
{
    auto update_count = *upd_count;

    for (csize_t i = 0; i < update_count; ++i)
    {
        auto idx = updated[i];

        auto work = count - idx;

        auto threads = ((work / 8 + warpSize - 1) / warpSize) * warpSize;
        auto blocks = (threads + max_threads - 1) / max_threads;
        threads = threads > max_threads ? max_threads : threads;
        blocks = blocks > max_blocks ? max_blocks : blocks;

        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);

        point_neighbor<N>
            <<<blocks, threads, shared_size, s>>>(centroids, neighbors, work_neighbors, dim, count, idx, new_idx);
        reduce_u<N><<<1, 32, 0, s>>>(work_neighbors, neighbors, idx, blocks);

        cudaStreamDestroy(s);
    }
}
