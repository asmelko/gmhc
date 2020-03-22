#include <kernels.cuh>

#include <device_launch_parameters.h>

using namespace clustering;

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

__global__ void print_a(asgn_t* assignments, size_t point_size)
{
	for (size_t i = 0; i < point_size; i++)
		printf("%d: %d\n", (int)i, (int)assignments[i]);
}

__global__ void print_kind(cluster_kind* kind, size_t count)
{
	for (size_t i = 0; i < count; i++)
	{
		printf("%d. %d\n", (int)i, (int)kind[i]);
	}
}


void run_print_assg(asgn_t* assignments, size_t point_size)
{
	print_a << <1, 1 >> > (assignments, point_size);
}

void run_print_kind(cluster_kind* kind, size_t count)
{
	print_kind << <1, 1 >> > (kind, count);
}