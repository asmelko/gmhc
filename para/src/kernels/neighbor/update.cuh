#include <kernels.cuh>

#include <device_launch_parameters.h>
#include "neighbor_common.cuh"

using namespace clustering;

template <csize_t N>
__global__ void update(neighbor_t* __restrict__  neighbors_a, csize_t* __restrict__ updated,
	csize_t* __restrict__ small_work_idx, csize_t* __restrict__ big_work_idx,
	csize_t small_size, csize_t big_begin, csize_t big_size, pasgn_t move_a, pasgn_t move_b, csize_t new_idx)
{
	for (csize_t idx = threadIdx.x + blockDim.x * blockIdx.x;
		idx < small_size + big_size;
		idx += blockDim.x * gridDim.x)
	{
		if (idx >= small_size)
			idx += big_begin - small_size;

		if (idx == move_a.first || idx == move_b.first || idx == new_idx)
		{
			csize_t store_idx = atomicAdd(idx < small_size ? small_work_idx : big_work_idx, 1);

			updated[store_idx] = idx;
			continue;
		}

		neighbor_t tmp_nei[N];
		memcpy(tmp_nei, neighbors_a + idx * N, sizeof(neighbor_t) * N);

		csize_t last_empty = 0;

		for (csize_t i = 0; i < N; i++)
		{
			if (tmp_nei[i].distance == FLT_MAX)
				break;

			if (tmp_nei[i].idx == move_a.first || tmp_nei[i].idx == move_b.first)
				tmp_nei[i].distance = FLT_MAX;
			else
			{
				if (tmp_nei[i].idx == move_a.second)
					tmp_nei[i].idx = move_a.first;
				else if (tmp_nei[i].idx == move_b.second)
					tmp_nei[i].idx = move_b.first;

				tmp_nei[last_empty++] = tmp_nei[i];
			}
		}

		for (csize_t i = last_empty; i < N; i++)
			tmp_nei[i].distance = FLT_MAX;

		if (tmp_nei[0].distance == FLT_MAX)
		{
			csize_t store_idx = atomicAdd(idx < small_size ? small_work_idx : big_work_idx, 1);

			updated[store_idx] = idx;
		}

		memcpy(neighbors_a + idx * N, tmp_nei, sizeof(neighbor_t) * N);

		if (idx >= small_size)
			idx -= big_begin - small_size;
	}
}