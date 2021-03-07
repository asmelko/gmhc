#ifndef NEIGHBOR_COMMON_CUH
#define NEIGHBOR_COMMON_CUH

#include <cfloat>
#include <device_launch_parameters.h>

#include "../common_kernels.cuh"

template<clustering::csize_t N>
__device__ __inline__ void add_neighbor(neighbor_t* __restrict__ neighbors, neighbor_t neighbor)
{
    neighbor_t prev_min;
    clustering::csize_t i = 0;
    for (; i < N; ++i)
    {
        if (neighbors[i].distance > neighbor.distance)
        {
            prev_min = neighbors[i];
            neighbors[i] = neighbor;
            break;
        }
    }

    for (++i; i < N; i++)
    {
        if (prev_min.distance == FLT_INF)
            return;

        neighbor_t tmp = neighbors[i];
        neighbors[i] = prev_min;
        prev_min = tmp;
    }
}

template<>
__device__ __inline__ void add_neighbor<1>(neighbor_t* __restrict__ neighbors, neighbor_t neighbor)
{
    if (neighbors->distance > neighbor.distance)
        *neighbors = neighbor;
}

template<clustering::csize_t N>
__device__ __inline__ void add_neighbor_disruptive(neighbor_t* __restrict__ neighbors, neighbor_t neighbor)
{
    neighbor_t prev_min;
    clustering::csize_t i = 0;
    for (; i < N; ++i)
    {
        if (neighbors[i].distance == FLT_INF)
            return;
        else if (neighbors[i].distance > neighbor.distance)
        {
            prev_min = neighbors[i];
            neighbors[i] = neighbor;
            break;
        }
    }

    for (++i; i < N; i++)
    {
        if (prev_min.distance == FLT_INF)
            return;

        neighbor_t tmp = neighbors[i];
        neighbors[i] = prev_min;
        prev_min = tmp;
    }
}

template<>
__device__ __inline__ void add_neighbor_disruptive<1>(neighbor_t* __restrict__ neighbors, neighbor_t neighbor)
{
    if (neighbors->distance != FLT_INF && neighbors->distance > neighbor.distance)
        *neighbors = neighbor;
}

template<clustering::csize_t N>
__device__ __inline__ void merge_neighbors(const neighbor_t* __restrict__ l_neighbors,
    const neighbor_t* __restrict__ r_neighbors,
    neighbor_t* __restrict__ res)
{
    clustering::csize_t l_idx = 0, r_idx = 0;

    for (clustering::csize_t i = 0; i < N; ++i)
    {
        if (l_neighbors[l_idx].distance < r_neighbors[r_idx].distance)
            res[i] = l_neighbors[l_idx++];
        else
            res[i] = r_neighbors[r_idx++];
    }
}

template<>
__device__ __inline__ void merge_neighbors<1>(const neighbor_t* __restrict__ l_neighbors,
    const neighbor_t* __restrict__ r_neighbors,
    neighbor_t* __restrict__ res)
{
    *res = l_neighbors->distance < r_neighbors->distance ? *l_neighbors : *r_neighbors;
}


template<clustering::csize_t N>
__device__ __inline__ void reduce_min_warp(neighbor_t* __restrict__ neighbors)
{
    for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        neighbor_t tmp[N];
        for (clustering::csize_t i = 0; i < N; ++i)
        {
            tmp[i].distance = __shfl_down_sync(0xFFFFFFFF, neighbors[i].distance, offset);
            tmp[i].idx = __shfl_down_sync(0xFFFFFFFF, neighbors[i].idx, offset);
        }

        neighbor_t tmp_cpy[N];
        merge_neighbors<N>(neighbors, tmp, tmp_cpy);
        memcpy(neighbors, tmp_cpy, sizeof(neighbor_t) * N);
    }
}

template<clustering::csize_t N>
__device__ __inline__ void reduce_min_block(
    neighbor_t* __restrict__ neighbors, neighbor_t* __restrict__ shared_mem, bool reduce_warp = true)
{
    if (reduce_warp)
        reduce_min_warp<N>(neighbors);

    auto lane_id = threadIdx.x % warpSize;
    auto warp_id = threadIdx.x / warpSize;

    if (lane_id == 0)
        memcpy(shared_mem + warp_id * N, neighbors, sizeof(neighbor_t) * N);

    __syncthreads();

    if (threadIdx.x < blockDim.x / warpSize)
        memcpy(neighbors, shared_mem + threadIdx.x * N, sizeof(neighbor_t) * N);
    else
        for (clustering::csize_t i = 0; i < N; i++)
            neighbors[i].distance = FLT_INF;

    reduce_min_warp<N>(neighbors);
}

#endif