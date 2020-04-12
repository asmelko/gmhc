#include <kernels.cuh>

#include <device_launch_parameters.h>

using namespace clustering;

__global__ void merge_clusters(asgn_t* __restrict__ assignments, csize_t point_size, asgn_t old_A, asgn_t old_B, asgn_t new_C)
{
	for (csize_t i = blockIdx.x * blockDim.x + threadIdx.x; i < point_size; i += gridDim.x * blockDim.x)
		if (assignments[i] == old_A || assignments[i] == old_B)
			assignments[i] = new_C;
}

void run_merge_clusters(asgn_t* assignments, csize_t point_size, asgn_t old_A, asgn_t old_B, asgn_t new_C, kernel_info info)
{
	merge_clusters<<<info.grid_dim, info.block_dim>>>(assignments, point_size, old_A, old_B, new_C);
}
