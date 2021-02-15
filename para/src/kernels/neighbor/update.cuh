#include <device_launch_parameters.h>

#include "kernels.cuh"
#include "neighbor_eucl.cuh"
#include "neighbor_maha.cuh"

using namespace clustering;

template<csize_t N>
__device__ __inline__ bool update_single_neighbors(
    neighbor_t* __restrict__ neighbors, csize_t size, csize_t old_a, csize_t old_b)
{
    neighbor_t tmp_nei[N];
    memcpy(tmp_nei, neighbors, sizeof(neighbor_t) * N);

    csize_t last_empty = 0;

    for (csize_t i = 0; i < N; i++)
    {
        if (tmp_nei[i].distance == FLT_INF)
            break;

        if (tmp_nei[i].idx == old_a || tmp_nei[i].idx == old_b)
            tmp_nei[i].distance = FLT_INF;
        else
        {
            if (tmp_nei[i].idx == size)
                tmp_nei[i].idx = old_b;

            tmp_nei[last_empty++] = tmp_nei[i];
        }
    }

    for (csize_t i = last_empty; i < N; i++)
        tmp_nei[i].distance = FLT_INF;

    memcpy(neighbors, tmp_nei, sizeof(neighbor_t) * N);

    return tmp_nei[0].distance == FLT_INF;
}

template<>
__device__ __inline__ bool update_single_neighbors<1>(
    neighbor_t* __restrict__ neighbors, csize_t size, csize_t old_a, csize_t old_b)
{
    neighbor_t tmp_nei = *neighbors;

    if (tmp_nei.distance == FLT_INF)
        return true;

    if (tmp_nei.idx == old_a || tmp_nei.idx == old_b)
    {
        tmp_nei.distance = FLT_INF;
        memcpy(neighbors, &tmp_nei, sizeof(neighbor_t));
    }
    else if (tmp_nei.idx == size)
    {
        tmp_nei.idx = old_b;
        memcpy(neighbors, &tmp_nei, sizeof(neighbor_t));
    }

    return tmp_nei.distance == FLT_INF;
}

struct neigh_invoke_info
{
    float* centroids;
    float* inverses;
    float* mfactors;
    neighbor_t* work_neighbors;
    csize_t dim;
    csize_t thread_limit;
    csize_t block_limit;
    bool use_eucl;
};

template<csize_t N>
__device__ __inline__ void compute_kernel_params(csize_t thread_limit,
    csize_t block_limit,
    bool use_eucl,
    csize_t work,
    csize_t dim,
    csize_t& threads,
    csize_t& blocks,
    csize_t& shared)
{
    constexpr size_t factor = 4;
    if (use_eucl)
    {
        auto tmp = (work + factor - 1) / factor;
        threads = ((tmp + warpSize - 1) / warpSize) * warpSize;
    }
    else
        threads = work * warpSize;
    blocks = (threads + thread_limit - 1) / thread_limit;
    threads = threads > thread_limit ? thread_limit : threads;
    blocks = blocks > block_limit ? block_limit : blocks;

    auto warps = threads / warpSize;
    auto shared_new = (dim + warps + 1) * dim * sizeof(float);
    shared = max(shared_new, warpSize * sizeof(neighbor_t) * N);
}

template<csize_t N>
__device__ __inline__ void invoke_neighbor_u(
    neigh_invoke_info& info, neighbor_t* neighbors, csize_t size, csize_t idx, csize_t new_idx)
{
    csize_t threads, blocks, shared;
    compute_kernel_params<N>(
        info.thread_limit, info.block_limit, info.use_eucl, size - idx, info.dim, threads, blocks, shared);

    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);

    if (info.use_eucl)
        point_neighbor<N><<<blocks, threads, shared, s>>>(
            info.centroids, neighbors, info.work_neighbors, info.dim, size, idx, new_idx);
    else
        point_neighbors_mat<N><<<blocks, threads, shared, s>>>(
            info.centroids, info.inverses, info.mfactors, neighbors, info.work_neighbors, info.dim, size, idx, new_idx);

    reduce_u<N><<<1, 32, 0, s>>>(info.work_neighbors, neighbors, idx, blocks);

    cudaStreamDestroy(s);
}

template<csize_t N>
__global__ void update(
    neighbor_t* __restrict__ neighbors_a, neigh_invoke_info neigh_info, csize_t size, csize_t old_a, csize_t old_b)
{
    for (csize_t idx = threadIdx.x + blockDim.x * blockIdx.x; idx < size; idx += blockDim.x * gridDim.x)
    {
        if (idx == old_a)
            continue;

        if (idx == old_b)
        {
            invoke_neighbor_u<N>(neigh_info, neighbors_a, size, idx, old_a);
            continue;
        }

        bool need_update = update_single_neighbors<N>(neighbors_a + idx * N, size, old_a, old_b);

        if (need_update)
            invoke_neighbor_u<N>(neigh_info, neighbors_a, size, idx, old_a);
    }
}