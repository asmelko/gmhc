#include <kernels.cuh>

#include <device_launch_parameters.h>
#include "neighbor_common.cuh"

using namespace clustering;

template <csize_t N>
__inline__ __device__ void point_reduce
(const neighbor_t* __restrict__ neighbors, csize_t to_reduce, neighbor_t* __restrict__ reduced, csize_t idx)
{
	csize_t block = idx;
	csize_t nei = threadIdx.x % warpSize;

	neighbor_t local[N];

	if (nei < to_reduce)
		memcpy(local, neighbors + (block * to_reduce + nei) * N, sizeof(neighbor_t) * N);
	else
		for (csize_t i = 0; i < N; i++)
			local[i].distance = FLT_MAX;


	for (nei += warpSize; nei < to_reduce; nei += warpSize)
	{
		neighbor_t tmp[N];
		merge_neighbors<N>(local, neighbors + (block * to_reduce + nei) * N, tmp);
		memcpy(local, tmp, sizeof(neighbor_t) * N);
	}

	reduce_min_warp<N>(local);

	if (threadIdx.x % warpSize == 0)
	{
		memcpy(reduced + block * N, local, sizeof(neighbor_t) * N);
	}
}


template <csize_t N>
__global__ void reduce(const neighbor_t* __restrict__ neighbors, neighbor_t* __restrict__ reduced,
	csize_t small_count, csize_t big_begin, csize_t big_count, csize_t to_reduce)
{
	csize_t idx = threadIdx.x + blockIdx.x * blockDim.x;

	for (; idx < small_count * warpSize; idx += blockDim.x * gridDim.x)
		point_reduce<N>(neighbors, to_reduce, reduced, idx / warpSize);

	idx += (big_begin - small_count) * warpSize;

	for (; idx < (big_begin + big_count) * warpSize; idx += blockDim.x * gridDim.x)
		point_reduce<N>(neighbors, to_reduce, reduced, idx / warpSize);
}

template <csize_t N>
__global__ void reduce_u(const neighbor_t* __restrict__ neighbors, neighbor_t* __restrict__ reduced,
	const csize_t* __restrict__ updated, const csize_t* __restrict__ eucl_upd_size,
	csize_t maha_begin, const csize_t* __restrict__ maha_upd_size, csize_t to_reduce)
{
	auto eucl_count = *eucl_upd_size;

	csize_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	for (; idx < eucl_count * warpSize; idx += blockDim.x * gridDim.x)
	{
		point_reduce<N>(neighbors, to_reduce, reduced, updated[idx / warpSize]);
	}

	auto maha_count = *maha_upd_size - maha_begin;

	idx += (maha_begin - eucl_count) * warpSize;

	for (; idx < (maha_begin + maha_count) * warpSize; idx += blockDim.x * gridDim.x)
	{
		point_reduce<N>(neighbors, to_reduce, reduced, updated[idx / warpSize]);
	}
}