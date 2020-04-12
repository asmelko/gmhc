#ifndef NEIGHBOR_COMMON_CUH
#define NEIGHBOR_COMMON_CUH

#include <device_launch_parameters.h>
#include <cfloat>

#include "../common_kernels.cuh"

template <size_t N>
__device__ void add_neighbour(neighbour_t* neighbours, neighbour_t neighbour)
{
	neighbour_t prev_min;
	size_t i = 0;
	for (; i < N; ++i)
	{
		if (neighbours[i].distance > neighbour.distance)
		{
			prev_min = neighbours[i];
			neighbours[i] = neighbour;
			break;
		}
	}

	for (++i; i < N; i++)
	{
		if (prev_min.distance == FLT_MAX)
			return;

		neighbour_t tmp = neighbours[i];
		neighbours[i] = prev_min;
		prev_min = tmp;
	}
}

template <size_t N>
__device__ void merge_neighbours(const neighbour_t* l_neighbours, const neighbour_t* r_neighbours, neighbour_t* res)
{
	size_t l_idx = 0, r_idx = 0;

	for (size_t i = 0; i < N; ++i)
	{
		if (l_neighbours[l_idx].distance < r_neighbours[r_idx].distance)
			res[i] = l_neighbours[l_idx++];
		else
			res[i] = r_neighbours[r_idx++];
	}
}


template <size_t N>
__device__ void reduce_min_warp(neighbour_t* neighbours)
{
	for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2)
	{
		neighbour_t tmp[N];
		for (size_t i = 0; i < N; ++i)
		{
			tmp[i].distance = __shfl_down_sync(0xFFFFFFFF, neighbours[i].distance, offset);
			tmp[i].idx = __shfl_down_sync(0xFFFFFFFF, neighbours[i].idx, offset);
		}

		neighbour_t tmp_cpy[N];
		merge_neighbours<N>(neighbours, tmp, tmp_cpy);
		memcpy(neighbours, tmp_cpy, sizeof(neighbour_t) * N);
	}
}

template <size_t N>
__device__ void reduce_min_block(neighbour_t* neighbours, neighbour_t* shared_mem, bool reduce_warp = true)
{
	if (reduce_warp)
		reduce_min_warp<N>(neighbours);

	auto lane_id = threadIdx.x % warpSize;
	auto warp_id = threadIdx.x / warpSize;

	if (lane_id == 0)
		memcpy(shared_mem + warp_id * N, neighbours, sizeof(neighbour_t) * N);

	__syncthreads();

	if (threadIdx.x < blockDim.x / warpSize)
		memcpy(neighbours, shared_mem + threadIdx.x * N, sizeof(neighbour_t) * N);
	else
		for (size_t i = 0; i < N; i++)
			neighbours[i].distance = FLT_MAX;

	reduce_min_warp<N>(neighbours);
}

#endif