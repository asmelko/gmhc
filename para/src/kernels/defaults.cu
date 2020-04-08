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

__global__ void set_default(float* icov_matrix, size_t size)
{
	for (size_t i = threadIdx.x; i < size; i += blockDim.x)
		icov_matrix[i] = 1;
}

void run_set_default_inverse(float* icov_matrix, size_t size)
{
	set_default << <1, size >> > (icov_matrix, size);
}
