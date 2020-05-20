#include "kernels.cuh"
#include "device_launch_parameters.h"
#include <cfloat>

using namespace clustering;

__global__ void set_default_asgn(asgn_t* __restrict__ asgns, csize_t size)
{
	for (csize_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x)
		asgns[i] = i;
}

void run_set_default_asgn(asgn_t* asgns, csize_t N)
{
	set_default_asgn << <50, 1024 >> > (asgns, N);
}

__global__ void set_default(float* __restrict__ icov_matrix, csize_t size)
{
	for (csize_t i = threadIdx.x; i < size*size; i += blockDim.x)
		if (i / size == i % size)
			icov_matrix[i] = 1;
		else
			icov_matrix[i] = 0;
}

void run_set_default_inverse(float* icov_matrix, csize_t size)
{
	set_default << <1, size*size >> > (icov_matrix, size);
}

__global__ void set_default_neigh(neighbor_t* neighbors, csize_t count)
{
	for (csize_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x)
		neighbors[i].distance = FLT_INF;
}

void run_set_default_neigh(neighbor_t* neighbors, csize_t count, kernel_info info)
{
	set_default_neigh<< <info.grid_dim, info.block_dim>> > (neighbors, count);
}