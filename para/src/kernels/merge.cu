#include <kernels.cuh>

#include <device_launch_parameters.h>

using namespace clustering;

__global__ void merge_clusters(asgn_t* __restrict__ assignments, csize_t* __restrict__ assignment_idxs, csize_t* __restrict__ idxs_size,
	csize_t point_size, asgn_t old_A, asgn_t old_B, asgn_t new_C)
{
	for (csize_t i = blockIdx.x * blockDim.x + threadIdx.x; i < point_size / 4; i += gridDim.x * blockDim.x)
	{
		csize_t new_idx[4];
		csize_t size = 0;
		uint4 data = ((uint4*)assignments)[i];

		if (data.x == old_A || data.x == old_B)
		{
			data.x = new_C;
			new_idx[size++] = i * 4;
		}
		if (data.y == old_A || data.y == old_B)
		{
			data.y = new_C;
			new_idx[size++] = i * 4 + 1;
		}
		if (data.z == old_A || data.z == old_B)
		{
			data.z = new_C;
			new_idx[size++] = i * 4 + 2;
		}
		if (data.w == old_A || data.w == old_B)
		{
			data.w = new_C;
			new_idx[size++] = i * 4 + 3;
		}

		if (size)
		{
			((uint4*)assignments)[i] = data;
			auto idx = atomicAdd(idxs_size, size);
			for (size_t j = 0; j < size; ++j)
				assignment_idxs[idx + j] = new_idx[j];
		}
	}

	if (blockIdx.x * blockDim.x + threadIdx.x == 0 && (point_size % 4) != 0)
		for (csize_t i = point_size % 4; i > 0; --i)
			if (assignments[point_size - i] == old_A || assignments[point_size - i] == old_B)
			{
				assignments[i] = new_C;
				auto idx = atomicAdd(idxs_size, (csize_t)1);
				assignment_idxs[idx] = point_size - i;
			}
}

void run_merge_clusters(asgn_t* assignments, csize_t* assignment_idxs, csize_t* idxs_size,
	csize_t point_size, asgn_t old_A, asgn_t old_B, asgn_t new_C, kernel_info info)
{
	CUCH(cudaMemset(idxs_size, 0, sizeof(csize_t)));
	merge_clusters << <info.grid_dim, info.block_dim >> > (assignments, assignment_idxs, idxs_size, point_size, old_A, old_B, new_C);
}
