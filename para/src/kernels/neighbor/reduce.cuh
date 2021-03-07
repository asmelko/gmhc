#include <device_launch_parameters.h>

#include "kernels.cuh"
#include "neighbor_common.cuh"

using namespace clustering;

template<csize_t N, csize_t Storage>
__inline__ __device__ void point_reduce(
    const neighbor_t* __restrict__ neighbors, csize_t to_reduce, neighbor_t* __restrict__ reduced, csize_t idx)
{
    csize_t block = idx;
    csize_t nei = threadIdx.x % warpSize;

    neighbor_t local[N];

    if (nei < to_reduce)
        memcpy(local, neighbors + (block * to_reduce + nei) * Storage, sizeof(neighbor_t) * N);
    else
        for (csize_t i = 0; i < N; i++)
            local[i].distance = FLT_INF;


    for (nei += warpSize; nei < to_reduce; nei += warpSize)
    {
        neighbor_t tmp[N];
        merge_neighbors<N>(local, neighbors + (block * to_reduce + nei) * Storage, tmp);
        memcpy(local, tmp, sizeof(neighbor_t) * N);
    }

    reduce_min_warp<N>(local);

    if (threadIdx.x % warpSize == 0)
    {
        memcpy(reduced + block * Storage, local, sizeof(neighbor_t) * N);
    }
}


template<csize_t N, csize_t Storage>
__global__ void reduce(
    const neighbor_t* __restrict__ neighbors, neighbor_t* __restrict__ reduced, csize_t count, csize_t to_reduce)
{
    for (csize_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < count * warpSize; idx += blockDim.x * gridDim.x)
        point_reduce<N, Storage>(neighbors, to_reduce, reduced, idx / warpSize);
}

template<csize_t N, csize_t Storage>
__global__ void reduce_u(const neighbor_t* __restrict__ neighbors,
    neighbor_t* __restrict__ reduced,
    const csize_t* __restrict__ updated,
    const csize_t* __restrict__ upd_size,
    csize_t to_reduce)
{
    auto count = *upd_size;

    for (csize_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < count * warpSize; idx += blockDim.x * gridDim.x)
        point_reduce<N, Storage>(neighbors, to_reduce, reduced, updated[idx / warpSize]);
}

template<csize_t N, csize_t Storage>
__global__ void reduce_u(
    const neighbor_t* __restrict__ neighbors, neighbor_t* __restrict__ reduced, csize_t new_idx, csize_t to_reduce)
{
    point_reduce<N, Storage>(neighbors, to_reduce, reduced, new_idx);
}