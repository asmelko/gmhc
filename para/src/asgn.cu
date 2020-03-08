#include <kernels.cuh>

#include <device_launch_parameters.h>

using namespace clustering;

__global__ void set_default_asgn(clustering::centroid_data_t* __restrict__ asgns, size_t N)
{
	for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
	{
		asgns[i].id = i;
		asgns[i].icov = nullptr;
	}
}

void run_set_default_asgn(clustering::centroid_data_t* asgns, size_t N)
{
	set_default_asgn << <50, 1024 >> > (asgns, N);
}

__global__ void set_default_asgn(clustering::asgn_t* __restrict__ asgns, size_t N)
{
	for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
		asgns[i] = i;
}

void run_set_default_asgn(clustering::asgn_t* asgns, size_t N)
{
	set_default_asgn << <50, 1024 >> > (asgns, N);
}

__global__ void merge_clusters(asgn_t* __restrict__ assignments, size_t point_size, asgn_t old_A, asgn_t old_B, asgn_t new_C)
{
	for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < point_size; i += gridDim.x * blockDim.x)
		if (assignments[i] == old_A || assignments[i] == old_B)
			assignments[i] = new_C;
}

void run_merge_clusters(asgn_t* assignments, size_t point_size, asgn_t old_A, asgn_t old_B, asgn_t new_C, kernel_info info)
{
	merge_clusters << <info.grid_dim, info.block_dim >> > (assignments, point_size, old_A, old_B, new_C);
}