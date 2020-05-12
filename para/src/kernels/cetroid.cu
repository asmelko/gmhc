#include <kernels.cuh>

#include <device_launch_parameters.h>

#include "common_kernels.cuh"

using namespace clustering;

__global__ void centroid(const float* __restrict__ points, csize_t dim, csize_t count, 
	const asgn_t* __restrict__ assignments, float* __restrict__ out, asgn_t cid, csize_t cluster_size)
{
	extern __shared__ float shared_mem[];

	float tmp[MAX_DIM];

	memset(tmp, 0, dim * sizeof(float));
	for (size_t i = threadIdx.x; i < dim; i+=blockDim.x)
	{
		shared_mem[i] = 0;
	}

	__syncthreads();

	for (csize_t idx = blockDim.x * blockIdx.x + threadIdx.x; idx < count; idx += gridDim.x * blockDim.x)
	{
		if (assignments[idx] == cid)
		{
			for (csize_t i = 0; i < dim; ++i)
				tmp[i] += points[idx * dim + i];
		}
	}

	//reduce_sum_block(tmp, dim, shared_mem);

	for (size_t i = 0; i < dim; i++)
	{
		atomicAdd(shared_mem+ i, tmp[i]);
	}

	__syncthreads();

	if (threadIdx.x == 0)
		for (csize_t i = 0; i < dim; ++i)
			atomicAdd(out + i, shared_mem[i]);
}

__global__ void set_mem(float* __restrict__ mem)
{
	mem[threadIdx.x] = 0;
}

__global__ void fin_mem(float* __restrict__ mem, csize_t div)
{
	mem[threadIdx.x] /= div;
}

void run_centroid(const input_t in, const asgn_t* assignments, float* out, asgn_t cetroid_id, csize_t cluster_size, kernel_info info)
{
	set_mem << <1, in.dim >> > (out);
	centroid << <1, 1024, 32 * (in.dim * sizeof(float)) >> > (in.data, in.dim, in.count, assignments, out, cetroid_id, cluster_size);
	fin_mem << < 1, in.dim >> > (out, cluster_size);

}
