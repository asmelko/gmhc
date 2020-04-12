#include <kernels.cuh>

#include <device_launch_parameters.h>
#include "neighbor_common.cuh"

using namespace clustering;

template <csize_t N>
__inline__ __device__ void point_reduce
(const neighbour_t* __restrict__ neighbours, csize_t to_reduce, neighbour_t* __restrict__ reduced, csize_t idx)
{
	csize_t block = idx;
	csize_t nei = threadIdx.x % warpSize;

	neighbour_t local[N];

	if (nei < to_reduce)
		memcpy(local, neighbours + (block * to_reduce + nei) * N, sizeof(neighbour_t) * N);
	else
		for (csize_t i = 0; i < N; i++)
			local[i].distance = FLT_MAX;


	for (nei += warpSize; nei < to_reduce; nei += warpSize)
	{
		neighbour_t tmp[N];
		merge_neighbours<N>(local, neighbours + (block * to_reduce + nei) * N, tmp);
		memcpy(local, tmp, sizeof(neighbour_t) * N);
	}

	reduce_min_warp<N>(local);

	if (threadIdx.x % warpSize == 0)
	{
		memcpy(reduced + block * N, local, sizeof(neighbour_t) * N);
	}
}


template <csize_t N>
__global__ void reduce(const neighbour_t* __restrict__ neighbours, csize_t to_reduce, csize_t count, neighbour_t* __restrict__  reduced)
{
	for (csize_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < count * warpSize; idx += blockDim.x * gridDim.x)
	{
		point_reduce<N>(neighbours, to_reduce, reduced, idx / warpSize);
	}
}

template <csize_t N>
__global__ void reduce_u(const neighbour_t* __restrict__ neighbours, neighbour_t* __restrict__ reduced,
	const csize_t* __restrict__ updated, const csize_t* __restrict__ eucl_upd_size,
	csize_t maha_begin, const csize_t* __restrict__ maha_upd_size, csize_t to_reduce)
{
	auto eucl_count = *eucl_upd_size;

	csize_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	for (; idx < eucl_count * warpSize; idx += blockDim.x * gridDim.x)
	{
		point_reduce<N>(neighbours, to_reduce, reduced, updated[idx / warpSize]);
	}

	auto maha_count = *maha_upd_size - maha_begin;

	idx += (maha_begin - eucl_count) * warpSize;

	for (; idx < (maha_begin + maha_count) * warpSize; idx += blockDim.x * gridDim.x)
	{
		point_reduce<N>(neighbours, to_reduce, reduced, updated[idx / warpSize]);
	}
}